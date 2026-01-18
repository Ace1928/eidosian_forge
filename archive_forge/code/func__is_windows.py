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
def _is_windows(path: str) -> bool:
    if len(path) < 3:
        return False
    return path[0].isalpha() and path[1] == ':' and (path[2] == '/')