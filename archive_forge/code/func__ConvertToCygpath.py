import errno
import os
import re
import subprocess
import sys
import glob
def _ConvertToCygpath(path):
    """Convert to cygwin path if we are using cygwin."""
    if sys.platform == 'cygwin':
        p = subprocess.Popen(['cygpath', path], stdout=subprocess.PIPE)
        path = p.communicate()[0].decode('utf-8').strip()
    return path