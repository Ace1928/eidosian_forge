from . import errors, registry
from .branch import Branch
from .repository import Repository
from .revision import Revision
class ForeignVcs:
    """A foreign version control system."""
    branch_format = None
    repository_format = None

    def __init__(self, mapping_registry, abbreviation=None):
        """Create a new foreign vcs instance.

        :param mapping_registry: Registry with mappings for this VCS.
        :param abbreviation: Optional abbreviation ('bzr', 'svn', 'git', etc)
        """
        self.abbreviation = abbreviation
        self.mapping_registry = mapping_registry

    def show_foreign_revid(self, foreign_revid):
        """Prepare a foreign revision id for formatting using bzr log.

        :param foreign_revid: Foreign revision id.
        :return: Dictionary mapping string keys to string values.
        """
        return {}

    def serialize_foreign_revid(self, foreign_revid):
        """Serialize a foreign revision id for this VCS.

        :param foreign_revid: Foreign revision id
        :return: Bytestring with serialized revid, will not contain any
            newlines.
        """
        raise NotImplementedError(self.serialize_foreign_revid)