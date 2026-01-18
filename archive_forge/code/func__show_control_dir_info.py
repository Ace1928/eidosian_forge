import sys
import time
from io import StringIO
from . import branch as _mod_branch
from . import controldir, errors
from . import hooks as _mod_hooks
from . import osutils, urlutils
from .bzr import bzrdir
from .errors import (NoRepositoryPresent, NotBranchError, NotLocalUrl,
from .missing import find_unmerged
def _show_control_dir_info(control, outfile):
    """Show control dir information."""
    if control._format.colocated_branches:
        outfile.write('\n')
        outfile.write('Control directory:\n')
        outfile.write('         %d branches\n' % len(control.list_branches()))