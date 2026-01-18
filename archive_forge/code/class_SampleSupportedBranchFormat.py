from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
class SampleSupportedBranchFormat(_mod_bzrbranch.BranchFormatMetadir):
    """A sample supported format."""

    @classmethod
    def get_format_string(cls):
        """See BzrBranchFormat.get_format_string()."""
        return SampleSupportedBranchFormatString

    def initialize(self, a_controldir, name=None, append_revisions_only=None):
        t = a_controldir.get_branch_transport(self, name=name)
        t.put_bytes('format', self.get_format_string())
        return 'A branch'

    def open(self, transport, name=None, _found=False, ignore_fallbacks=False, possible_transports=None):
        return 'opened supported branch.'