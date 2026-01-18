from collections import defaultdict
import binascii
from io import BytesIO, UnsupportedOperation
from collections import (
import difflib
import struct
from itertools import chain
import os
import sys
from hashlib import sha1
from os import (
from struct import unpack_from
import zlib
from dulwich.errors import (  # noqa: E402
from dulwich.file import GitFile  # noqa: E402
from dulwich.lru_cache import (  # noqa: E402
from dulwich.objects import (  # noqa: E402
def _iter_unpacked(self):
    self._file.seek(self._header_size)
    if self._num_objects is None:
        return
    for _ in range(self._num_objects):
        offset = self._file.tell()
        unpacked, unused = unpack_object(self._file.read, compute_crc32=False)
        unpacked.offset = offset
        yield unpacked
        self._file.seek(-len(unused), SEEK_CUR)