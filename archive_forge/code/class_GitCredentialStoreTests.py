import base64
import os
import shutil
import sys
import tempfile
import warnings
from io import BytesIO
from typing import Dict
from unittest.mock import patch
from urllib.parse import quote as urlquote
from urllib.parse import urlparse
import dulwich
from dulwich import client
from dulwich.tests import TestCase, skipIf
from ..client import (
from ..config import ConfigDict
from ..objects import Commit, Tree
from ..pack import pack_objects_to_data, write_pack_data, write_pack_objects
from ..protocol import TCP_GIT_PORT, Protocol
from ..repo import MemoryRepo, Repo
from .utils import open_repo, setup_warning_catcher, tear_down_repo
class GitCredentialStoreTests(TestCase):

    @classmethod
    def setUpClass(cls):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b'https://user:pass@example.org\n')
        cls.fname = f.name

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.fname)

    def test_nonmatching_scheme(self):
        self.assertEqual(get_credentials_from_store(b'http', b'example.org', fnames=[self.fname]), None)

    def test_nonmatching_hostname(self):
        self.assertEqual(get_credentials_from_store(b'https', b'noentry.org', fnames=[self.fname]), None)

    def test_match_without_username(self):
        self.assertEqual(get_credentials_from_store(b'https', b'example.org', fnames=[self.fname]), (b'user', b'pass'))

    def test_match_with_matching_username(self):
        self.assertEqual(get_credentials_from_store(b'https', b'example.org', b'user', fnames=[self.fname]), (b'user', b'pass'))

    def test_no_match_with_nonmatching_username(self):
        self.assertEqual(get_credentials_from_store(b'https', b'example.org', b'otheruser', fnames=[self.fname]), None)