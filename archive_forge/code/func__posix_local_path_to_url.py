import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def _posix_local_path_to_url(path):
    """Convert a local path like ./foo into a URL like file:///path/to/foo

    This also handles transforming escaping unicode characters, etc.
    """
    return 'file://' + escape(posixpath.abspath(path))