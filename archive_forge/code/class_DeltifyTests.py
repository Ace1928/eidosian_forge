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
class DeltifyTests(TestCase):

    def test_empty(self):
        self.assertEqual([], list(deltify_pack_objects([])))

    def test_single(self):
        b = Blob.from_string(b'foo')
        self.assertEqual([UnpackedObject(b.type_num, sha=b.sha().digest(), delta_base=None, decomp_chunks=b.as_raw_chunks())], list(deltify_pack_objects([(b, b'')])))

    def test_simple_delta(self):
        b1 = Blob.from_string(b'a' * 101)
        b2 = Blob.from_string(b'a' * 100)
        delta = list(create_delta(b1.as_raw_chunks(), b2.as_raw_chunks()))
        self.assertEqual([UnpackedObject(b1.type_num, sha=b1.sha().digest(), delta_base=None, decomp_chunks=b1.as_raw_chunks()), UnpackedObject(b2.type_num, sha=b2.sha().digest(), delta_base=b1.sha().digest(), decomp_chunks=delta)], list(deltify_pack_objects([(b1, b''), (b2, b'')])))