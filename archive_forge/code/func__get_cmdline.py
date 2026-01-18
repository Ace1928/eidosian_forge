import io
import os
import re
import sys
from ._core import Process
def _get_cmdline(pid):
    path = os.path.join('/proc', str(pid), 'cmdline')
    encoding = sys.getfilesystemencoding() or 'utf-8'
    with io.open(path, encoding=encoding, errors='replace') as f:
        return tuple(f.read().split('\x00')[:-1])