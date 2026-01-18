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
def _show_format_info(control=None, repository=None, branch=None, working=None, outfile=None):
    """Show known formats for control, working, branch and repository."""
    outfile.write('\n')
    outfile.write('Format:\n')
    if control:
        outfile.write('       control: %s\n' % control._format.get_format_description())
    if working:
        outfile.write('  working tree: %s\n' % working._format.get_format_description())
    if branch:
        outfile.write('        branch: %s\n' % branch._format.get_format_description())
    if repository:
        outfile.write('    repository: %s\n' % repository._format.get_format_description())