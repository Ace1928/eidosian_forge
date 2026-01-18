from breezy.lazy_import import lazy_import
from ... import config, merge
import fnmatch
import subprocess
import tempfile
from breezy import (
def _invoke(self, command):
    trace.mutter('Will msgmerge: {}'.format(command))
    proc = subprocess.Popen(cmdline.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    out, err = proc.communicate()
    return (proc.returncode, out, err)