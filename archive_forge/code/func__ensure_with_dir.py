import glob
import os
import struct
from .lazy_import import lazy_import
import ctypes
from breezy import cmdline
from breezy.i18n import gettext
def _ensure_with_dir(path):
    if not os.path.split(path)[0] or path.startswith('*') or path.startswith('?'):
        return ('./' + path, True)
    else:
        return (path, False)