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
class _FSPath:

    def __init__(self, path: str):
        if path is None:
            raise ValueError("path can't be None")
        path = _modify_path(path)
        self._is_windows = False
        if _is_windows(path):
            self._scheme = ''
            self._root = path[:3]
            self._path = path[3:]
            self._is_windows = True
        elif path.startswith('/'):
            self._scheme = ''
            self._root = '/'
            self._path = fs.path.abspath(path)
        else:
            uri = urlparse(path)
            if uri.scheme == '' and (not path.startswith('/')):
                raise ValueError(f'invalid {path}, must be abs path either local or with scheme')
            self._scheme = uri.scheme
            if uri.netloc == '':
                raise ValueError(f'invalid path {path}')
            self._root = uri.scheme + '://' + uri.netloc
            self._path = uri.path
        self._path = self._path.lstrip('/')

    @property
    def is_windows(self) -> bool:
        return self._is_windows

    @property
    def scheme(self) -> str:
        return self._scheme

    @property
    def root(self) -> str:
        return self._root

    @property
    def relative_path(self) -> str:
        return self._path