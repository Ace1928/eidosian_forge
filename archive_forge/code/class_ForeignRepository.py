from . import errors, registry
from .branch import Branch
from .repository import Repository
from .revision import Revision
class ForeignRepository(Repository):
    """A Repository that exists in a foreign version control system.

    The data in this repository can not be represented natively using
    Bazaars internal datastructures, but have to converted using a VcsMapping.
    """
    vcs: ForeignVcs

    def has_foreign_revision(self, foreign_revid):
        """Check whether the specified foreign revision is present.

        :param foreign_revid: A foreign revision id, in the format used
                              by this Repository's VCS.
        """
        raise NotImplementedError(self.has_foreign_revision)

    def lookup_bzr_revision_id(self, revid):
        """Lookup a mapped or roundtripped revision by revision id.

        :param revid: Bazaar revision id
        :return: Tuple with foreign revision id and mapping.
        """
        raise NotImplementedError(self.lookup_revision_id)

    def all_revision_ids(self, mapping=None):
        """See Repository.all_revision_ids()."""
        raise NotImplementedError(self.all_revision_ids)

    def get_default_mapping(self):
        """Get the default mapping for this repository."""
        raise NotImplementedError(self.get_default_mapping)