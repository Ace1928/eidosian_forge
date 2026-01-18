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
class _BoundGlobber(BoundGlobber):

    def __call__(self, pattern: Any, path: str='/', namespaces: Any=None, case_sensitive: bool=True, exclude_dirs: Any=None) -> Globber:
        fp = _FSPath(path)
        _path = fs.path.join(fp._root, fp._path) if fp.is_windows else path
        return super().__call__(pattern, path=_path, namespaces=namespaces, case_sensitive=case_sensitive, exclude_dirs=exclude_dirs)