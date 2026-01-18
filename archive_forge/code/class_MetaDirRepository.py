import itertools
from .. import errors, lockable_files, lockdir
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from ..repository import Repository, RepositoryFormat, format_registry
from . import bzrdir
class MetaDirRepository(Repository):
    """Repositories in the new meta-dir layout.

    :ivar _transport: Transport for access to repository control files,
        typically pointing to .bzr/repository.
    """
    _format: 'RepositoryFormatMetaDir'

    def __init__(self, _format, a_bzrdir, control_files):
        super().__init__(_format, a_bzrdir, control_files)
        self._transport = control_files._transport

    def is_shared(self):
        """Return True if this repository is flagged as a shared repository."""
        return self._transport.has('shared-storage')

    def set_make_working_trees(self, new_value):
        """Set the policy flag for making working trees when creating branches.

        This only applies to branches that use this repository.

        The default is 'True'.
        :param new_value: True to restore the default, False to disable making
                          working trees.
        """
        with self.lock_write():
            if new_value:
                try:
                    self._transport.delete('no-working-trees')
                except _mod_transport.NoSuchFile:
                    pass
            else:
                self._transport.put_bytes('no-working-trees', b'', mode=self.controldir._get_file_mode())

    def make_working_trees(self):
        """Returns the policy for making working trees on new branches."""
        return not self._transport.has('no-working-trees')

    def update_feature_flags(self, updated_flags):
        """Update the feature flags for this branch.

        :param updated_flags: Dictionary mapping feature names to necessities
            A necessity can be None to indicate the feature should be removed
        """
        with self.lock_write():
            self._format._update_feature_flags(updated_flags)
            self.control_transport.put_bytes('format', self._format.as_string())

    def _find_parent_ids_of_revisions(self, revision_ids):
        """Find all parent ids that are mentioned in the revision graph.

        :return: set of revisions that are parents of revision_ids which are
            not part of revision_ids themselves
        """
        parent_ids = set(itertools.chain.from_iterable(self.get_parent_map(revision_ids).values()))
        parent_ids.difference_update(revision_ids)
        parent_ids.discard(_mod_revision.NULL_REVISION)
        return parent_ids