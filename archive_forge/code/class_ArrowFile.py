import errno
import io
import os
import secrets
import shutil
from contextlib import suppress
from functools import cached_property, wraps
from urllib.parse import parse_qs
from fsspec.spec import AbstractFileSystem
from fsspec.utils import (
@mirror_from('stream', ['read', 'seek', 'tell', 'write', 'readable', 'writable', 'close', 'size', 'seekable'])
class ArrowFile(io.IOBase):

    def __init__(self, fs, stream, path, mode, block_size=None, **kwargs):
        self.path = path
        self.mode = mode
        self.fs = fs
        self.stream = stream
        self.blocksize = self.block_size = block_size
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.close()