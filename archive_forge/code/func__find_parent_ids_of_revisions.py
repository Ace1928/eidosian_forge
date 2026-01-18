import itertools
from .. import errors, lockable_files, lockdir
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from ..repository import Repository, RepositoryFormat, format_registry
from . import bzrdir
def _find_parent_ids_of_revisions(self, revision_ids):
    """Find all parent ids that are mentioned in the revision graph.

        :return: set of revisions that are parents of revision_ids which are
            not part of revision_ids themselves
        """
    parent_ids = set(itertools.chain.from_iterable(self.get_parent_map(revision_ids).values()))
    parent_ids.difference_update(revision_ids)
    parent_ids.discard(_mod_revision.NULL_REVISION)
    return parent_ids