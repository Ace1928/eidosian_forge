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
class DeltaEncodeSizeTests(TestCase):

    def test_basic(self):
        self.assertEqual(b'\x00', _delta_encode_size(0))
        self.assertEqual(b'\x01', _delta_encode_size(1))
        self.assertEqual(b'\xfa\x01', _delta_encode_size(250))
        self.assertEqual(b'\xe8\x07', _delta_encode_size(1000))
        self.assertEqual(b'\xa0\x8d\x06', _delta_encode_size(100000))