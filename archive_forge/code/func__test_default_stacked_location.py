from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def _test_default_stacked_location(self):
    branch = self.make_branch('a', format=self.get_format_name())
    self.assertRaises(errors.NotStacked, branch.get_stacked_on_url)