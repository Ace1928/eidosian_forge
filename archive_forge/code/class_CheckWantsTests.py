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
class CheckWantsTests(TestCase):

    def test_fine(self):
        check_wants([b'2f3dc7a53fb752a6961d3a56683df46d4d3bf262'], {b'refs/heads/blah': b'2f3dc7a53fb752a6961d3a56683df46d4d3bf262'})

    def test_missing(self):
        self.assertRaises(InvalidWants, check_wants, [b'2f3dc7a53fb752a6961d3a56683df46d4d3bf262'], {b'refs/heads/blah': b'3f3dc7a53fb752a6961d3a56683df46d4d3bf262'})

    def test_annotated(self):
        self.assertRaises(InvalidWants, check_wants, [b'2f3dc7a53fb752a6961d3a56683df46d4d3bf262'], {b'refs/heads/blah': b'3f3dc7a53fb752a6961d3a56683df46d4d3bf262', b'refs/heads/blah^{}': b'2f3dc7a53fb752a6961d3a56683df46d4d3bf262'})