from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils
from ... import revision as _mod_revision
from ...bzr import inventory
from ...bzr.inventorytree import InventoryTreeChange
class RevisionStore:

    def __init__(self, repo):
        """An object responsible for loading revisions into a repository.

        NOTE: Repository locking is not managed by this class. Clients
        should take a write lock, call load() multiple times, then release
        the lock.

        :param repository: the target repository
        """
        self.repo = repo
        self._graph = None
        self._use_known_graph = True
        self._supports_chks = getattr(repo._format, 'supports_chks', False)

    def expects_rich_root(self):
        """Does this store expect inventories with rich roots?"""
        return self.repo.supports_rich_root()

    def init_inventory(self, revision_id):
        """Generate an inventory for a parentless revision."""
        if self._supports_chks:
            inv = self._init_chk_inventory(revision_id, inventory.ROOT_ID)
        else:
            inv = inventory.Inventory(revision_id=revision_id)
            if self.expects_rich_root():
                inv.root.revision = revision_id
        return inv

    def _init_chk_inventory(self, revision_id, root_id):
        """Generate a CHKInventory for a parentless revision."""
        from ...bzr import chk_map
        chk_store = self.repo.chk_bytes
        serializer = self.repo._format._serializer
        search_key_name = serializer.search_key_name
        maximum_size = serializer.maximum_size
        inv = inventory.CHKInventory(search_key_name)
        inv.revision_id = revision_id
        inv.root_id = root_id
        search_key_func = chk_map.search_key_registry.get(search_key_name)
        inv.id_to_entry = chk_map.CHKMap(chk_store, None, search_key_func)
        inv.id_to_entry._root_node.set_maximum_size(maximum_size)
        inv.parent_id_basename_to_file_id = chk_map.CHKMap(chk_store, None, search_key_func)
        inv.parent_id_basename_to_file_id._root_node.set_maximum_size(maximum_size)
        inv.parent_id_basename_to_file_id._root_node._key_width = 2
        return inv

    def get_inventory(self, revision_id):
        """Get a stored inventory."""
        return self.repo.get_inventory(revision_id)

    def get_file_lines(self, revision_id, path):
        """Get the lines stored for a file in a given revision."""
        revtree = self.repo.revision_tree(revision_id)
        return revtree.get_file_lines(path)

    def start_new_revision(self, revision, parents, parent_invs):
        """Init the metadata needed for get_parents_and_revision_for_entry().

        :param revision: a Revision object
        """
        self._current_rev_id = revision.revision_id
        self._rev_parents = parents
        self._rev_parent_invs = parent_invs
        config = None
        self._commit_builder = self.repo.get_commit_builder(self.repo, parents, config, timestamp=revision.timestamp, timezone=revision.timezone, committer=revision.committer, revprops=revision.properties, revision_id=revision.revision_id)

    def get_parents_and_revision_for_entry(self, ie):
        """Get the parents and revision for an inventory entry.

        :param ie: the inventory entry
        :return parents, revision_id where
            parents is the tuple of parent revision_ids for the per-file graph
            revision_id is the revision_id to use for this entry
        """
        if self._current_rev_id is None:
            raise AssertionError('start_new_revision() must be called before get_parents_and_revision_for_entry()')
        if ie.revision != self._current_rev_id:
            raise AssertionError('start_new_revision() registered a different revision (%s) to that in the inventory entry (%s)' % (self._current_rev_id, ie.revision))
        parent_candidate_entries = ie.parent_candidates(self._rev_parent_invs)
        head_set = self._commit_builder._heads(ie.file_id, list(parent_candidate_entries))
        heads = []
        for inv in self._rev_parent_invs:
            try:
                old_rev = inv.get_entry(ie.file_id).revision
            except errors.NoSuchId:
                pass
            else:
                if old_rev in head_set:
                    rev_id = inv.get_entry(ie.file_id).revision
                    heads.append(rev_id)
                    head_set.remove(rev_id)
        if len(heads) == 0:
            return ((), ie.revision)
        parent_entry = parent_candidate_entries[heads[0]]
        changed = False
        if len(heads) > 1:
            changed = True
        elif parent_entry.name != ie.name or parent_entry.kind != ie.kind or parent_entry.parent_id != ie.parent_id:
            changed = True
        elif ie.kind == 'file':
            if parent_entry.text_sha1 != ie.text_sha1 or parent_entry.executable != ie.executable:
                changed = True
        elif ie.kind == 'symlink':
            if parent_entry.symlink_target != ie.symlink_target:
                changed = True
        if changed:
            rev_id = ie.revision
        else:
            rev_id = parent_entry.revision
        return (tuple(heads), rev_id)

    def load_using_delta(self, rev, basis_inv, inv_delta, signature, text_provider, parents_provider, inventories_provider=None):
        """Load a revision by applying a delta to a (CHK)Inventory.

        :param rev: the Revision
        :param basis_inv: the basis Inventory or CHKInventory
        :param inv_delta: the inventory delta
        :param signature: signing information
        :param text_provider: a callable expecting a file_id parameter
            that returns the text for that file-id
        :param parents_provider: a callable expecting a file_id parameter
            that return the list of parent-ids for that file-id
        :param inventories_provider: a callable expecting a repository and
            a list of revision-ids, that returns:
              * the list of revision-ids present in the repository
              * the list of inventories for the revision-id's,
                including an empty inventory for the missing revisions
            If None, a default implementation is provided.
        """
        builder = self.repo._commit_builder_class(self.repo, parents=rev.parent_ids, config=None, timestamp=rev.timestamp, timezone=rev.timezone, committer=rev.committer, revprops=rev.properties, revision_id=rev.revision_id)
        if self._graph is None and self._use_known_graph:
            if getattr(_mod_graph, 'GraphThunkIdsToKeys', None) and getattr(_mod_graph.GraphThunkIdsToKeys, 'add_node', None) and getattr(self.repo, 'get_known_graph_ancestry', None):
                self._graph = self.repo.get_known_graph_ancestry(rev.parent_ids)
            else:
                self._use_known_graph = False
        if self._graph is not None:
            orig_heads = builder._heads

            def thunked_heads(file_id, revision_ids):
                if len(revision_ids) < 2:
                    res = set(revision_ids)
                else:
                    res = set(self._graph.heads(revision_ids))
                return res
            builder._heads = thunked_heads
        if rev.parent_ids:
            basis_rev_id = rev.parent_ids[0]
        else:
            basis_rev_id = _mod_revision.NULL_REVISION
        tree = _TreeShim(self.repo, basis_inv, inv_delta, text_provider)
        changes = tree._delta_to_iter_changes()
        for path, fs_hash in builder.record_iter_changes(tree, basis_rev_id, changes):
            pass
        builder.finish_inventory()
        if isinstance(builder.inv_sha1, tuple):
            builder.inv_sha1, builder.new_inventory = builder.inv_sha1
        rev.inv_sha1 = builder.inv_sha1
        config = builder._config_stack
        builder.repository.add_revision(builder._new_revision_id, rev, builder.revision_tree().root_inventory)
        if self._graph is not None:
            self._graph.add_node(builder._new_revision_id, rev.parent_ids)
        if signature is not None:
            raise AssertionError('signatures not guaranteed yet')
            self.repo.add_signature_text(rev.revision_id, signature)
        return builder.revision_tree().root_inventory