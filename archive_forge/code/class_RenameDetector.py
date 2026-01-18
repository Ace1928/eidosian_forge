import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
class RenameDetector:
    """Object for handling rename detection between two trees."""

    def __init__(self, store, rename_threshold=RENAME_THRESHOLD, max_files=MAX_FILES, rewrite_threshold=REWRITE_THRESHOLD, find_copies_harder=False) -> None:
        """Initialize the rename detector.

        Args:
          store: An ObjectStore for looking up objects.
          rename_threshold: The threshold similarity score for considering
            an add/delete pair to be a rename/copy; see _similarity_score.
          max_files: The maximum number of adds and deletes to consider,
            or None for no limit. The detector is guaranteed to compare no more
            than max_files ** 2 add/delete pairs. This limit is provided
            because rename detection can be quadratic in the project size. If
            the limit is exceeded, no content rename detection is attempted.
          rewrite_threshold: The threshold similarity score below which a
            modify should be considered a delete/add, or None to not break
            modifies; see _similarity_score.
          find_copies_harder: If True, consider unmodified files when
            detecting copies.
        """
        self._store = store
        self._rename_threshold = rename_threshold
        self._rewrite_threshold = rewrite_threshold
        self._max_files = max_files
        self._find_copies_harder = find_copies_harder
        self._want_unchanged = False

    def _reset(self):
        self._adds = []
        self._deletes = []
        self._changes = []

    def _should_split(self, change):
        if self._rewrite_threshold is None or change.type != CHANGE_MODIFY or change.old.sha == change.new.sha:
            return False
        old_obj = self._store[change.old.sha]
        new_obj = self._store[change.new.sha]
        return _similarity_score(old_obj, new_obj) < self._rewrite_threshold

    def _add_change(self, change):
        if change.type == CHANGE_ADD:
            self._adds.append(change)
        elif change.type == CHANGE_DELETE:
            self._deletes.append(change)
        elif self._should_split(change):
            self._deletes.append(TreeChange.delete(change.old))
            self._adds.append(TreeChange.add(change.new))
        elif self._find_copies_harder and change.type == CHANGE_UNCHANGED or change.type == CHANGE_MODIFY:
            self._deletes.append(change)
        else:
            self._changes.append(change)

    def _collect_changes(self, tree1_id, tree2_id):
        want_unchanged = self._find_copies_harder or self._want_unchanged
        for change in tree_changes(self._store, tree1_id, tree2_id, want_unchanged=want_unchanged, include_trees=self._include_trees):
            self._add_change(change)

    def _prune(self, add_paths, delete_paths):
        self._adds = [a for a in self._adds if a.new.path not in add_paths]
        self._deletes = [d for d in self._deletes if d.old.path not in delete_paths]

    def _find_exact_renames(self):
        add_map = defaultdict(list)
        for add in self._adds:
            add_map[add.new.sha].append(add.new)
        delete_map = defaultdict(list)
        for delete in self._deletes:
            is_delete = delete.type == CHANGE_DELETE
            delete_map[delete.old.sha].append((delete.old, is_delete))
        add_paths = set()
        delete_paths = set()
        for sha, sha_deletes in delete_map.items():
            sha_adds = add_map[sha]
            for (old, is_delete), new in zip(sha_deletes, sha_adds):
                if stat.S_IFMT(old.mode) != stat.S_IFMT(new.mode):
                    continue
                if is_delete:
                    delete_paths.add(old.path)
                add_paths.add(new.path)
                new_type = is_delete and CHANGE_RENAME or CHANGE_COPY
                self._changes.append(TreeChange(new_type, old, new))
            num_extra_adds = len(sha_adds) - len(sha_deletes)
            old = sha_deletes[0][0]
            if num_extra_adds > 0:
                for new in sha_adds[-num_extra_adds:]:
                    add_paths.add(new.path)
                    self._changes.append(TreeChange(CHANGE_COPY, old, new))
        self._prune(add_paths, delete_paths)

    def _should_find_content_renames(self):
        return len(self._adds) * len(self._deletes) <= self._max_files ** 2

    def _rename_type(self, check_paths, delete, add):
        if check_paths and delete.old.path == add.new.path:
            return CHANGE_MODIFY
        elif delete.type != CHANGE_DELETE:
            return CHANGE_COPY
        return CHANGE_RENAME

    def _find_content_rename_candidates(self):
        candidates = self._candidates = []
        if not self._should_find_content_renames():
            return
        block_cache = {}
        check_paths = self._rename_threshold is not None
        for delete in self._deletes:
            if S_ISGITLINK(delete.old.mode):
                continue
            old_sha = delete.old.sha
            old_obj = self._store[old_sha]
            block_cache[old_sha] = _count_blocks(old_obj)
            for add in self._adds:
                if stat.S_IFMT(delete.old.mode) != stat.S_IFMT(add.new.mode):
                    continue
                new_obj = self._store[add.new.sha]
                score = _similarity_score(old_obj, new_obj, block_cache=block_cache)
                if score > self._rename_threshold:
                    new_type = self._rename_type(check_paths, delete, add)
                    rename = TreeChange(new_type, delete.old, add.new)
                    candidates.append((-score, rename))

    def _choose_content_renames(self):
        self._candidates.sort()
        delete_paths = set()
        add_paths = set()
        for _, change in self._candidates:
            new_path = change.new.path
            if new_path in add_paths:
                continue
            old_path = change.old.path
            orig_type = change.type
            if old_path in delete_paths:
                change = TreeChange(CHANGE_COPY, change.old, change.new)
            if orig_type != CHANGE_COPY:
                delete_paths.add(old_path)
            add_paths.add(new_path)
            self._changes.append(change)
        self._prune(add_paths, delete_paths)

    def _join_modifies(self):
        if self._rewrite_threshold is None:
            return
        modifies = {}
        delete_map = {d.old.path: d for d in self._deletes}
        for add in self._adds:
            path = add.new.path
            delete = delete_map.get(path)
            if delete is not None and stat.S_IFMT(delete.old.mode) == stat.S_IFMT(add.new.mode):
                modifies[path] = TreeChange(CHANGE_MODIFY, delete.old, add.new)
        self._adds = [a for a in self._adds if a.new.path not in modifies]
        self._deletes = [a for a in self._deletes if a.new.path not in modifies]
        self._changes += modifies.values()

    def _sorted_changes(self):
        result = []
        result.extend(self._adds)
        result.extend(self._deletes)
        result.extend(self._changes)
        result.sort(key=_tree_change_key)
        return result

    def _prune_unchanged(self):
        if self._want_unchanged:
            return
        self._deletes = [d for d in self._deletes if d.type != CHANGE_UNCHANGED]

    def changes_with_renames(self, tree1_id, tree2_id, want_unchanged=False, include_trees=False):
        """Iterate TreeChanges between two tree SHAs, with rename detection."""
        self._reset()
        self._want_unchanged = want_unchanged
        self._include_trees = include_trees
        self._collect_changes(tree1_id, tree2_id)
        self._find_exact_renames()
        self._find_content_rename_candidates()
        self._choose_content_renames()
        self._join_modifies()
        self._prune_unchanged()
        return self._sorted_changes()