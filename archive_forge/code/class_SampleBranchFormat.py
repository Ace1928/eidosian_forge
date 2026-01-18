from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
class SampleBranchFormat(_mod_bzrbranch.BranchFormatMetadir):
    """A sample format

    this format is initializable, unsupported to aid in testing the
    open and open_downlevel routines.
    """

    @classmethod
    def get_format_string(cls):
        """See BzrBranchFormat.get_format_string()."""
        return b'Sample branch format.'

    def initialize(self, a_controldir, name=None, repository=None, append_revisions_only=None):
        """Format 4 branches cannot be created."""
        t = a_controldir.get_branch_transport(self, name=name)
        t.put_bytes('format', self.get_format_string())
        return 'A branch'

    def is_supported(self):
        return False

    def open(self, transport, name=None, _found=False, ignore_fallbacks=False, possible_transports=None):
        return 'opened branch.'