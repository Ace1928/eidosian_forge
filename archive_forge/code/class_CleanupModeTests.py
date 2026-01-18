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
class CleanupModeTests(TestCase):

    def assertModeEqual(self, expected, got):
        self.assertEqual(expected, got, f'{expected:o} != {got:o}')

    def test_file(self):
        self.assertModeEqual(33188, cleanup_mode(32768))

    def test_executable(self):
        self.assertModeEqual(33261, cleanup_mode(33225))
        self.assertModeEqual(33261, cleanup_mode(33216))

    def test_symlink(self):
        self.assertModeEqual(40960, cleanup_mode(41417))

    def test_dir(self):
        self.assertModeEqual(16384, cleanup_mode(16729))

    def test_submodule(self):
        self.assertModeEqual(57344, cleanup_mode(57828))