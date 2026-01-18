import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def _win32_local_path_from_url(url):
    """Convert a url like file:///C:/path/to/foo into C:/path/to/foo"""
    if not url.startswith('file://'):
        raise InvalidURL(url, 'local urls must start with file:///, UNC path urls must start with file://')
    url = strip_segment_parameters(url)
    win32_url = url[len('file:'):]
    if not win32_url.startswith('///'):
        if win32_url[2] == '/' or win32_url[3] in '|:':
            raise InvalidURL(url, 'Win32 UNC path urls have form file://HOST/path')
        return unescape(win32_url)
    if win32_url == '///':
        return '/'
    if len(win32_url) < 6 or win32_url[3] not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' or win32_url[4] not in '|:' or (win32_url[5] != '/'):
        raise InvalidURL(url, 'Win32 file urls start with file:///x:/, where x is a valid drive letter')
    return win32_url[3].upper() + ':' + unescape(win32_url[5:])