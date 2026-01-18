import os
import shutil
import sys
import tempfile
from io import BytesIO
from typing import Dict, List
from dulwich.tests import TestCase
from ..errors import (
from ..object_store import MemoryObjectStore
from ..objects import Tree
from ..protocol import ZERO_SHA, format_capability_line
from ..repo import MemoryRepo, Repo
from ..server import (
from .utils import make_commit, make_tag
class FileSystemBackendTests(TestCase):
    """Tests for FileSystemBackend."""

    def setUp(self):
        super().setUp()
        self.path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.path)
        self.repo = Repo.init(self.path)
        if sys.platform == 'win32':
            self.backend = FileSystemBackend(self.path[0] + ':' + os.sep)
        else:
            self.backend = FileSystemBackend()

    def test_nonexistant(self):
        self.assertRaises(NotGitRepository, self.backend.open_repository, '/does/not/exist/unless/foo')

    def test_absolute(self):
        repo = self.backend.open_repository(self.path)
        self.assertTrue(os.path.samefile(os.path.abspath(repo.path), os.path.abspath(self.repo.path)))

    def test_child(self):
        self.assertRaises(NotGitRepository, self.backend.open_repository, os.path.join(self.path, 'foo'))

    def test_bad_repo_path(self):
        backend = FileSystemBackend()
        self.assertRaises(NotGitRepository, lambda: backend.open_repository('/ups'))