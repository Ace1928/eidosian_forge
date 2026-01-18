from binascii import b2a_hex
from pathlib import Path
from .util import pack, unpack
from git.objects import Blob
from typing import NamedTuple, Sequence, TYPE_CHECKING, Tuple, Union, cast, List
from git.types import PathLike
class BaseIndexEntryHelper(NamedTuple):
    """Typed namedtuple to provide named attribute access for BaseIndexEntry.
    Needed to allow overriding __new__ in child class to preserve backwards compat."""
    mode: int
    binsha: bytes
    flags: int
    path: PathLike
    ctime_bytes: bytes = pack('>LL', 0, 0)
    mtime_bytes: bytes = pack('>LL', 0, 0)
    dev: int = 0
    inode: int = 0
    uid: int = 0
    gid: int = 0
    size: int = 0