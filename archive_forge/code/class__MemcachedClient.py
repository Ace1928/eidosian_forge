import errno
import fnmatch
import marshal
import os
import pickle
import stat
import sys
import tempfile
import typing as t
from hashlib import sha1
from io import BytesIO
from types import CodeType
class _MemcachedClient(te.Protocol):

    def get(self, key: str) -> bytes:
        ...

    def set(self, key: str, value: bytes, timeout: t.Optional[int]=None) -> None:
        ...