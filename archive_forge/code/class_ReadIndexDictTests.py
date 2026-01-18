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
class ReadIndexDictTests(IndexTestCase):

    def setUp(self):
        IndexTestCase.setUp(self)
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        IndexTestCase.tearDown(self)
        shutil.rmtree(self.tempdir)

    def test_simple_write(self):
        entries = {b'barbla': IndexEntry((1230680220, 0), (1230680220, 0), 2050, 3761020, 33188, 1000, 1000, 0, b'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391')}
        filename = os.path.join(self.tempdir, 'test-simple-write-index')
        with open(filename, 'wb+') as x:
            write_index_dict(x, entries)
        with open(filename, 'rb') as x:
            self.assertEqual(entries, read_index_dict(x))