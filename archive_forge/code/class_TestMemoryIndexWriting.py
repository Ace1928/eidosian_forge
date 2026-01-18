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
class TestMemoryIndexWriting(TestCase, BaseTestPackIndexWriting):

    def setUp(self):
        TestCase.setUp(self)
        self._has_crc32_checksum = True
        self._supports_large = True

    def index(self, filename, entries, pack_checksum):
        return MemoryPackIndex(entries, pack_checksum)

    def tearDown(self):
        TestCase.tearDown(self)