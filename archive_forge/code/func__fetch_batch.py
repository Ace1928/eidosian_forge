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