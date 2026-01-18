from . import errors, registry
from .branch import Branch
from .repository import Repository
from .revision import Revision
class VcsMapping:
    """Describes the mapping between the semantics of Bazaar and a foreign VCS.

    """
    experimental = False
    roundtripping = False
    revid_prefix: bytes

    def __init__(self, vcs):
        """Create a new VcsMapping.

        :param vcs: VCS that this mapping maps to Bazaar
        """
        self.vcs = vcs

    def revision_id_bzr_to_foreign(self, bzr_revid):
        """Parse a bzr revision id and convert it to a foreign revid.

        :param bzr_revid: The bzr revision id (a string).
        :return: A foreign revision id, can be any sort of object.
        """
        raise NotImplementedError(self.revision_id_bzr_to_foreign)

    def revision_id_foreign_to_bzr(self, foreign_revid):
        """Parse a foreign revision id and convert it to a bzr revid.

        :param foreign_revid: Foreign revision id, can be any sort of object.
        :return: A bzr revision id.
        """
        raise NotImplementedError(self.revision_id_foreign_to_bzr)