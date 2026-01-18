import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def _posix_local_path_from_url(url):
    """Convert a url like file:///path/to/foo into /path/to/foo"""
    url = strip_segment_parameters(url)
    file_localhost_prefix = 'file://localhost/'
    if url.startswith(file_localhost_prefix):
        path = url[len(file_localhost_prefix) - 1:]
    elif not url.startswith('file:///'):
        raise InvalidURL(url, 'local urls must start with file:/// or file://localhost/')
    else:
        path = url[len('file://'):]
    return unescape(path)