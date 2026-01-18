import os
import shutil
import sys
import tempfile
import zlib
from hashlib import sha1
from io import BytesIO
from typing import Set
from dulwich.tests import TestCase
from ..errors import ApplyDeltaError, ChecksumMismatch
from ..file import GitFile
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit, Tree, hex_to_sha, sha_to_hex
from ..pack import (
from .utils import build_pack, make_object
def _resolve_object(self, offset, pack_type_num, base_chunks):
    assert offset not in self._unpacked_offsets, 'Attempted to re-inflate offset %i' % offset
    self._unpacked_offsets.add(offset)
    return super()._resolve_object(offset, pack_type_num, base_chunks)