import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def _win32_local_path_to_url(path):
    """Convert a local path like ./foo into a URL like file:///C:/path/to/foo

    This also handles transforming escaping unicode characters, etc.
    """
    if path == '/':
        return 'file:///'
    win32_path = osutils._win32_abspath(path)
    if win32_path.startswith('//'):
        return 'file:' + escape(win32_path)
    return 'file:///' + str(win32_path[0].upper()) + ':' + escape(win32_path[2:])