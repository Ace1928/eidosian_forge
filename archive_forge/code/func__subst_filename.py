import os
import shutil
import subprocess
import sys
import tempfile
from .lazy_import import lazy_import
from breezy import (
def _subst_filename(args, filename):
    subst_names = {'base': filename + '.BASE', 'this': filename + '.THIS', 'other': filename + '.OTHER', 'result': filename}
    tmp_file = None
    subst_args = []
    for arg in args:
        if '{this_temp}' in arg and 'this_temp' not in subst_names:
            fh, tmp_file = tempfile.mkstemp('_bzr_mergetools_%s.THIS' % os.path.basename(filename))
            trace.mutter('fh=%r, tmp_file=%r', fh, tmp_file)
            os.close(fh)
            shutil.copy(filename + '.THIS', tmp_file)
            subst_names['this_temp'] = tmp_file
        arg = _format_arg(arg, subst_names)
        subst_args.append(arg)
    return (subst_args, tmp_file)