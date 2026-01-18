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
class BaseTestFilePackIndexWriting(BaseTestPackIndexWriting):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def index(self, filename, entries, pack_checksum):
        path = os.path.join(self.tempdir, filename)
        self.writeIndex(path, entries, pack_checksum)
        idx = load_pack_index(path)
        self.assertSucceeds(idx.check)
        self.assertEqual(idx.version, self._expected_version)
        return idx

    def writeIndex(self, filename, entries, pack_checksum):
        with GitFile(filename, 'wb') as f:
            self._write_fn(f, entries, pack_checksum)