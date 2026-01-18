import binascii
import os
import posixpath
import stat
import warnings
import zlib
from collections import namedtuple
from hashlib import sha1
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
@staticmethod
def _parse_legacy_object_header(magic, f: BinaryIO) -> 'ShaFile':
    """Parse a legacy object, creating it but not reading the file."""
    bufsize = 1024
    decomp = zlib.decompressobj()
    header = decomp.decompress(magic)
    start = 0
    end = -1
    while end < 0:
        extra = f.read(bufsize)
        header += decomp.decompress(extra)
        magic += extra
        end = header.find(b'\x00', start)
        start = len(header)
    header = header[:end]
    type_name, size = header.split(b' ', 1)
    try:
        int(size)
    except ValueError as exc:
        raise ObjectFormatException('Object size not an integer: %s' % exc) from exc
    obj_class = object_class(type_name)
    if not obj_class:
        raise ObjectFormatException('Not a known type: %s' % type_name.decode('ascii'))
    return obj_class()