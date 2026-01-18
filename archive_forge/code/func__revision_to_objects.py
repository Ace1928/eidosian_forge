import posixpath
import stat
from typing import Dict, Iterable, Iterator, List
from dulwich.object_store import BaseObjectStore
from dulwich.objects import (ZERO_SHA, Blob, Commit, ObjectID, ShaFile, Tree,
from dulwich.pack import Pack, PackData, pack_objects_to_data
from .. import errors, lru_cache, osutils, trace, ui
from ..bzr.testament import StrictTestament3
from ..lock import LogicalLockResult
from ..revision import NULL_REVISION
from ..tree import InterTree
from .cache import from_repository as cache_from_repository
from .mapping import (default_mapping, encode_git_path, entry_mode,
from .unpeel_map import UnpeelMap
def _revision_to_objects(self, rev, tree, lossy, add_cache_entry=None):
    """Convert a revision to a set of git objects.

        :param rev: Bazaar revision object
        :param tree: Bazaar revision tree
        :param lossy: Whether to not roundtrip all Bazaar revision data
        """
    unusual_modes = extract_unusual_modes(rev)
    present_parents = self.repository.has_revisions(rev.parent_ids)
    parent_trees = self.tree_cache.revision_trees([p for p in rev.parent_ids if p in present_parents])
    root_tree = None
    for path, obj, bzr_key_data in _tree_to_objects(tree, parent_trees, self._cache.idmap, unusual_modes, self.mapping.BZR_DUMMY_FILE, add_cache_entry):
        if path == '':
            root_tree = obj
            root_key_data = bzr_key_data
        else:
            yield (path, obj)
    if root_tree is None:
        if not rev.parent_ids:
            root_tree = Tree()
        else:
            base_sha1 = self._lookup_revision_sha1(rev.parent_ids[0])
            root_tree = self[self[base_sha1].tree]
        root_key_data = (tree.path2id(''), tree.get_revision_id())
    if add_cache_entry is not None:
        add_cache_entry(root_tree, root_key_data, '')
    yield ('', root_tree)
    if not lossy:
        testament3 = StrictTestament3(rev, tree)
        verifiers = {'testament3-sha1': testament3.as_sha1()}
    else:
        verifiers = {}
    commit_obj = self._reconstruct_commit(rev, root_tree.id, lossy=lossy, verifiers=verifiers)
    try:
        foreign_revid, mapping = mapping_registry.parse_revision_id(rev.revision_id)
    except errors.InvalidRevisionId:
        pass
    else:
        _check_expected_sha(foreign_revid, commit_obj)
    if add_cache_entry is not None:
        add_cache_entry(commit_obj, verifiers, None)
    yield (None, commit_obj)