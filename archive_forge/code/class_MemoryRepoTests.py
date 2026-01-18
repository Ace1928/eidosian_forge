import glob
import locale
import os
import shutil
import stat
import sys
import tempfile
import warnings
from dulwich import errors, objects, porcelain
from dulwich.tests import TestCase, skipIf
from ..config import Config
from ..errors import NotGitRepository
from ..object_store import tree_lookup_path
from ..repo import (
from .utils import open_repo, setup_warning_catcher, tear_down_repo
import sys
from dulwich.repo import Repo
class MemoryRepoTests(TestCase):

    def test_set_description(self):
        r = MemoryRepo.init_bare([], {})
        description = b'Some description'
        r.set_description(description)
        self.assertEqual(description, r.get_description())

    def test_pull_into(self):
        r = MemoryRepo.init_bare([], {})
        repo = open_repo('a.git')
        self.addCleanup(tear_down_repo, repo)
        repo.fetch(r)