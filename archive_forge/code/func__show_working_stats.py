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
def _show_working_stats(working, outfile):
    """Show statistics about a working tree."""
    basis = working.basis_tree()
    delta = working.changes_from(basis, want_unchanged=True)
    outfile.write('\n')
    outfile.write('In the working tree:\n')
    outfile.write('  %8s unchanged\n' % len(delta.unchanged))
    outfile.write('  %8d modified\n' % len(delta.modified))
    outfile.write('  %8d added\n' % len(delta.added))
    outfile.write('  %8d removed\n' % len(delta.removed))
    outfile.write('  %8d renamed\n' % len(delta.renamed))
    outfile.write('  %8d copied\n' % len(delta.copied))
    ignore_cnt = unknown_cnt = 0
    for path in working.extras():
        if working.is_ignored(path):
            ignore_cnt += 1
        else:
            unknown_cnt += 1
    outfile.write('  %8d unknown\n' % unknown_cnt)
    outfile.write('  %8d ignored\n' % ignore_cnt)
    dir_cnt = 0
    for path, entry in working.iter_entries_by_dir():
        if entry.kind == 'directory' and path != '':
            dir_cnt += 1
    outfile.write('  %8d versioned %s\n' % (dir_cnt, plural(dir_cnt, 'subdirectory', 'subdirectories')))