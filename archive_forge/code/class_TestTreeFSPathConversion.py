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
class TestTreeFSPathConversion(TestCase):

    def test_tree_to_fs_path(self):
        tree_path = 'délwíçh/foo'.encode()
        fs_path = _tree_to_fs_path(b'/prefix/path', tree_path)
        self.assertEqual(fs_path, os.fsencode(os.path.join('/prefix/path', 'délwíçh', 'foo')))

    def test_fs_to_tree_path_str(self):
        fs_path = os.path.join(os.path.join('délwíçh', 'foo'))
        tree_path = _fs_to_tree_path(fs_path)
        self.assertEqual(tree_path, 'délwíçh/foo'.encode())

    def test_fs_to_tree_path_bytes(self):
        fs_path = os.path.join(os.fsencode(os.path.join('délwíçh', 'foo')))
        tree_path = _fs_to_tree_path(fs_path)
        self.assertEqual(tree_path, 'délwíçh/foo'.encode())