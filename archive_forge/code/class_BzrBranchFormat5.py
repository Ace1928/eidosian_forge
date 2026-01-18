from .. import debug, errors
from .. import revision as _mod_revision
from ..branch import Branch
from ..trace import mutter_callsite
from .branch import BranchFormatMetadir, BzrBranch
class BzrBranchFormat5(BranchFormatMetadir):
    """Bzr branch format 5.

    This format has:
     - a revision-history file.
     - a format string
     - a lock dir guarding the branch itself
     - all of this stored in a branch/ subdirectory
     - works with shared repositories.

    This format is new in bzr 0.8.
    """

    def _branch_class(self):
        return BzrBranch5

    @classmethod
    def get_format_string(cls):
        """See BranchFormat.get_format_string()."""
        return b'Bazaar-NG branch format 5\n'

    def get_format_description(self):
        """See BranchFormat.get_format_description()."""
        return 'Branch format 5'

    def initialize(self, a_controldir, name=None, repository=None, append_revisions_only=None):
        """Create a branch of this format in a_controldir."""
        if append_revisions_only:
            raise errors.UpgradeRequired(a_controldir.user_url)
        utf8_files = [('revision-history', b''), ('branch-name', b'')]
        return self._initialize_helper(a_controldir, utf8_files, name, repository)

    def supports_tags(self):
        return False
    supports_reference_locations = False