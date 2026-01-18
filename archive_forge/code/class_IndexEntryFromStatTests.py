import os
import shutil
import stat
import struct
import sys
import tempfile
from io import BytesIO
from dulwich.tests import TestCase, skipIf
from ..index import (
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..repo import Repo
class IndexEntryFromStatTests(TestCase):

    def test_simple(self):
        st = os.stat_result((16877, 131078, 64769, 154, 1000, 1000, 12288, 1323629595, 1324180496, 1324180496))
        entry = index_entry_from_stat(st, b'22' * 20)
        self.assertEqual(entry, IndexEntry(1324180496, 1324180496, 64769, 131078, 16384, 1000, 1000, 12288, b'2222222222222222222222222222222222222222'))

    def test_override_mode(self):
        st = os.stat_result((stat.S_IFREG + 420, 131078, 64769, 154, 1000, 1000, 12288, 1323629595, 1324180496, 1324180496))
        entry = index_entry_from_stat(st, b'22' * 20, mode=stat.S_IFREG + 493)
        self.assertEqual(entry, IndexEntry(1324180496, 1324180496, 64769, 131078, 33261, 1000, 1000, 12288, b'2222222222222222222222222222222222222222'))