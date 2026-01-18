from io import BytesIO
from ..lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import serializer
from breezy.i18n import gettext
from breezy.bzr.testament import Testament
from .. import errors
from ..decorators import only_raises
from ..repository import (CommitBuilder, FetchResult, InterRepository,
from ..trace import mutter, note
from .inventory import ROOT_ID, Inventory, entry_factory
from .inventorytree import InventoryTreeChange
from .repository import MetaDirRepository, RepositoryFormatMetaDir
class InterDifferingSerializer(InterVersionedFileRepository):

    @classmethod
    def _get_repo_format_to_test(self):
        return None

    @staticmethod
    def is_compatible(source, target):
        if not source._format.supports_full_versioned_files:
            return False
        if not target._format.supports_full_versioned_files:
            return False
        if 'IDS_never' in debug.debug_flags:
            return False
        if source.supports_rich_root() and (not target.supports_rich_root()):
            return False
        if source._format.supports_tree_reference and (not target._format.supports_tree_reference):
            return False
        if target._fallback_repositories and target._format.supports_chks:
            return False
        if 'IDS_always' in debug.debug_flags:
            return True
        if not source.controldir.transport.base.startswith('file:///'):
            return False
        if not target.controldir.transport.base.startswith('file:///'):
            return False
        return True

    def _get_trees(self, revision_ids, cache):
        possible_trees = []
        for rev_id in revision_ids:
            if rev_id in cache:
                possible_trees.append((rev_id, cache[rev_id]))
            else:
                try:
                    tree = self.source.revision_tree(rev_id)
                except errors.NoSuchRevision:
                    pass
                else:
                    cache[rev_id] = tree
                    possible_trees.append((rev_id, tree))
        return possible_trees

    def _get_delta_for_revision(self, tree, parent_ids, possible_trees):
        """Get the best delta and base for this revision.

        :return: (basis_id, delta)
        """
        deltas = []
        texts_possibly_new_in_tree = set()
        for basis_id, basis_tree in possible_trees:
            delta = tree.root_inventory._make_delta(basis_tree.root_inventory)
            for old_path, new_path, file_id, new_entry in delta:
                if new_path is None:
                    continue
                if not new_path:
                    continue
                kind = new_entry.kind
                if kind != 'directory' and kind != 'file':
                    continue
                texts_possibly_new_in_tree.add((file_id, new_entry.revision))
            deltas.append((len(delta), basis_id, delta))
        deltas.sort()
        return deltas[0][1:]

    def _fetch_parent_invs_for_stacking(self, parent_map, cache):
        """Find all parent revisions that are absent, but for which the
        inventory is present, and copy those inventories.

        This is necessary to preserve correctness when the source is stacked
        without fallbacks configured.  (Note that in cases like upgrade the
        source may be not have _fallback_repositories even though it is
        stacked.)
        """
        parent_revs = set(itertools.chain.from_iterable(parent_map.values()))
        present_parents = self.source.get_parent_map(parent_revs)
        absent_parents = parent_revs.difference(present_parents)
        parent_invs_keys_for_stacking = self.source.inventories.get_parent_map(((rev_id,) for rev_id in absent_parents))
        parent_inv_ids = [key[-1] for key in parent_invs_keys_for_stacking]
        for parent_tree in self.source.revision_trees(parent_inv_ids):
            current_revision_id = parent_tree.get_revision_id()
            parents_parents_keys = parent_invs_keys_for_stacking[current_revision_id,]
            parents_parents = [key[-1] for key in parents_parents_keys]
            basis_id = _mod_revision.NULL_REVISION
            basis_tree = self.source.revision_tree(basis_id)
            delta = parent_tree.root_inventory._make_delta(basis_tree.root_inventory)
            self.target.add_inventory_by_delta(basis_id, delta, current_revision_id, parents_parents)
            cache[current_revision_id] = parent_tree

    def _fetch_batch(self, revision_ids, basis_id, cache):
        """Fetch across a few revisions.

        :param revision_ids: The revisions to copy
        :param basis_id: The revision_id of a tree that must be in cache, used
            as a basis for delta when no other base is available
        :param cache: A cache of RevisionTrees that we can use.
        :return: The revision_id of the last converted tree. The RevisionTree
            for it will be in cache
        """
        root_keys_to_create = set()
        text_keys = set()
        pending_deltas = []
        pending_revisions = []
        parent_map = self.source.get_parent_map(revision_ids)
        self._fetch_parent_invs_for_stacking(parent_map, cache)
        self.source._safe_to_return_from_cache = True
        for tree in self.source.revision_trees(revision_ids):
            current_revision_id = tree.get_revision_id()
            parent_ids = parent_map.get(current_revision_id, ())
            parent_trees = self._get_trees(parent_ids, cache)
            possible_trees = list(parent_trees)
            if len(possible_trees) == 0:
                possible_trees.append((basis_id, cache[basis_id]))
            basis_id, delta = self._get_delta_for_revision(tree, parent_ids, possible_trees)
            revision = self.source.get_revision(current_revision_id)
            pending_deltas.append((basis_id, delta, current_revision_id, revision.parent_ids))
            if self._converting_to_rich_root:
                self._revision_id_to_root_id[current_revision_id] = tree.path2id('')
            texts_possibly_new_in_tree = set()
            for old_path, new_path, file_id, entry in delta:
                if new_path is None:
                    continue
                if not new_path:
                    if not self.target.supports_rich_root():
                        continue
                    if self._converting_to_rich_root:
                        root_keys_to_create.add((file_id, entry.revision))
                        continue
                kind = entry.kind
                texts_possibly_new_in_tree.add((file_id, entry.revision))
            for basis_id, basis_tree in possible_trees:
                basis_inv = basis_tree.root_inventory
                for file_key in list(texts_possibly_new_in_tree):
                    file_id, file_revision = file_key
                    try:
                        entry = basis_inv.get_entry(file_id)
                    except errors.NoSuchId:
                        continue
                    if entry.revision == file_revision:
                        texts_possibly_new_in_tree.remove(file_key)
            text_keys.update(texts_possibly_new_in_tree)
            pending_revisions.append(revision)
            cache[current_revision_id] = tree
            basis_id = current_revision_id
        self.source._safe_to_return_from_cache = False
        from_texts = self.source.texts
        to_texts = self.target.texts
        if root_keys_to_create:
            root_stream = _mod_fetch._new_root_data_stream(root_keys_to_create, self._revision_id_to_root_id, parent_map, self.source)
            to_texts.insert_record_stream(root_stream)
        to_texts.insert_record_stream(from_texts.get_record_stream(text_keys, self.target._format._fetch_order, not self.target._format._fetch_uses_deltas))
        for delta in pending_deltas:
            self.target.add_inventory_by_delta(*delta)
        if self.target._fallback_repositories:
            parent_ids = set()
            revision_ids = set()
            for revision in pending_revisions:
                revision_ids.add(revision.revision_id)
                parent_ids.update(revision.parent_ids)
            parent_ids.difference_update(revision_ids)
            parent_ids.discard(_mod_revision.NULL_REVISION)
            parent_map = self.source.get_parent_map(parent_ids)
            for parent_tree in self.source.revision_trees(parent_map):
                current_revision_id = parent_tree.get_revision_id()
                parents_parents = parent_map[current_revision_id]
                possible_trees = self._get_trees(parents_parents, cache)
                if len(possible_trees) == 0:
                    possible_trees.append((basis_id, cache[basis_id]))
                basis_id, delta = self._get_delta_for_revision(parent_tree, parents_parents, possible_trees)
                self.target.add_inventory_by_delta(basis_id, delta, current_revision_id, parents_parents)
        for revision in pending_revisions:
            try:
                signature = self.source.get_signature_text(revision.revision_id)
                self.target.add_signature_text(revision.revision_id, signature)
            except errors.NoSuchRevision:
                pass
            self.target.add_revision(revision.revision_id, revision)
        return basis_id

    def _fetch_all_revisions(self, revision_ids, pb):
        """Fetch everything for the list of revisions.

        :param revision_ids: The list of revisions to fetch. Must be in
            topological order.
        :param pb: A ProgressTask
        :return: None
        """
        basis_id, basis_tree = self._get_basis(revision_ids[0])
        batch_size = 100
        cache = lru_cache.LRUCache(100)
        cache[basis_id] = basis_tree
        del basis_tree
        hints = []
        a_graph = None
        for offset in range(0, len(revision_ids), batch_size):
            self.target.start_write_group()
            try:
                pb.update(gettext('Transferring revisions'), offset, len(revision_ids))
                batch = revision_ids[offset:offset + batch_size]
                basis_id = self._fetch_batch(batch, basis_id, cache)
            except:
                self.source._safe_to_return_from_cache = False
                self.target.abort_write_group()
                raise
            else:
                hint = self.target.commit_write_group()
                if hint:
                    hints.extend(hint)
        if hints and self.target._format.pack_compresses:
            self.target.pack(hint=hints)
        pb.update(gettext('Transferring revisions'), len(revision_ids), len(revision_ids))

    def fetch(self, revision_id=None, find_ghosts=False, fetch_spec=None, lossy=False):
        """See InterRepository.fetch()."""
        if lossy:
            raise errors.LossyPushToSameVCS(self.source, self.target)
        if fetch_spec is not None:
            revision_ids = fetch_spec.get_keys()
        else:
            revision_ids = None
        if self.source._format.experimental:
            ui.ui_factory.show_user_warning('experimental_format_fetch', from_format=self.source._format, to_format=self.target._format)
        if not self.source.supports_rich_root() and self.target.supports_rich_root():
            self._converting_to_rich_root = True
            self._revision_id_to_root_id = {}
        else:
            self._converting_to_rich_root = False
        if self.source._format.network_name() != self.target._format.network_name():
            ui.ui_factory.show_user_warning('cross_format_fetch', from_format=self.source._format, to_format=self.target._format)
        with self.lock_write():
            if revision_ids is None:
                if revision_id:
                    search_revision_ids = [revision_id]
                else:
                    search_revision_ids = None
                revision_ids = self.target.search_missing_revision_ids(self.source, revision_ids=search_revision_ids, find_ghosts=find_ghosts).get_keys()
            if not revision_ids:
                return FetchResult(0)
            revision_ids = tsort.topo_sort(self.source.get_graph().get_parent_map(revision_ids))
            if not revision_ids:
                return FetchResult(0)
            with ui.ui_factory.nested_progress_bar() as pb:
                self._fetch_all_revisions(revision_ids, pb)
            return FetchResult(len(revision_ids))

    def _get_basis(self, first_revision_id):
        """Get a revision and tree which exists in the target.

        This assumes that first_revision_id is selected for transmission
        because all other ancestors are already present. If we can't find an
        ancestor we fall back to NULL_REVISION since we know that is safe.

        :return: (basis_id, basis_tree)
        """
        first_rev = self.source.get_revision(first_revision_id)
        try:
            basis_id = first_rev.parent_ids[0]
            self.target.get_revision(basis_id)
            basis_tree = self.source.revision_tree(basis_id)
        except (IndexError, errors.NoSuchRevision):
            basis_id = _mod_revision.NULL_REVISION
            basis_tree = self.source.revision_tree(basis_id)
        return (basis_id, basis_tree)