from . import iterablefile, lock, revision, tree
class RevisionTree(tree.Tree):
    """Tree viewing a previous revision.

    File text can be retrieved from the text store.
    """

    def __init__(self, repository, revision_id):
        self._repository = repository
        self._revision_id = revision_id
        self._rules_searcher = None

    def has_versioned_directories(self):
        """See `Tree.has_versioned_directories`."""
        return self._repository._format.supports_versioned_directories

    def supports_tree_reference(self):
        return getattr(self._repository._format, 'supports_tree_reference', False)

    def get_parent_ids(self):
        """See Tree.get_parent_ids.

        A RevisionTree's parents match the revision graph.
        """
        if self._revision_id in (None, revision.NULL_REVISION):
            parent_ids = []
        else:
            parent_ids = self._repository.get_revision(self._revision_id).parent_ids
        return parent_ids

    def get_revision_id(self):
        """Return the revision id associated with this tree."""
        return self._revision_id

    def get_file_revision(self, path):
        """Return the revision id in which a file was last changed."""
        raise NotImplementedError(self.get_file_revision)

    def get_file_text(self, path):
        for identifier, content in self.iter_files_bytes([(path, None)]):
            return b''.join(content)

    def get_file(self, path):
        for identifier, content in self.iter_files_bytes([(path, None)]):
            return iterablefile.IterableFile(content)

    def is_locked(self):
        return self._repository.is_locked()

    def lock_read(self):
        self._repository.lock_read()
        return lock.LogicalLockResult(self.unlock)

    def __repr__(self):
        return '<{} instance at {:x}, rev_id={!r}>'.format(self.__class__.__name__, id(self), self._revision_id)

    def unlock(self):
        self._repository.unlock()

    def _get_rules_searcher(self, default_searcher):
        """See Tree._get_rules_searcher."""
        if self._rules_searcher is None:
            self._rules_searcher = super()._get_rules_searcher(default_searcher)
        return self._rules_searcher