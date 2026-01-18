import re
from threading import RLock
from typing import Any, Dict, Tuple
from urllib.parse import urlparse
from triad.utils.hash import to_uuid
import fs
from fs import memoryfs, open_fs, tempfs
from fs.base import FS as FSBase
from fs.glob import BoundGlobber, Globber
from fs.mountfs import MountFS
from fs.subfs import SubFS
def _modify_path(path: str) -> str:
    """to fix things like /s3:/a/b.txt -> s3://a/b.txt"""
    if path.startswith('/'):
        s = _SCHEME_PREFIX.search(path[1:])
        if s is not None:
            colon = s.end()
            scheme = path[1:colon]
            if colon + 1 == len(path):
                path = scheme + '://'
            elif path[colon + 1] == '/':
                path = scheme + '://' + path[colon + 1:].lstrip('/')
            elif path[colon + 1] == '\\':
                path = scheme + ':\\' + path[colon + 1:].lstrip('\\')
    if path.startswith('file:///'):
        path = path[8:]
    elif path.startswith('file://'):
        path = path[6:]
    if path.startswith('\\\\'):
        raise NotImplementedError(f'path {path} is not supported')
    if path != '' and path[0].isalpha():
        if len(path) == 2 and path[1] == ':':
            return path[0] + ':/'
        if path[1:].startswith(':\\'):
            return path[0] + ':/' + path[3:].replace('\\', '/').lstrip('/')
        if path[1:].startswith(':/'):
            return path[0] + ':/' + path[3:].lstrip('/')
    return path