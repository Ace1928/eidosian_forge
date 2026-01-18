import os
from http.client import parse_headers
from xmlrpc.client import Fault
import breezy
from ... import debug, tests, transport, urlutils
from ...branch import Branch
from ...directory_service import directories
from ...tests import (TestCaseInTempDir, TestCaseWithMemoryTransport, features,
from . import _register_directory
from .account import get_lp_login, set_lp_login
from .lp_directory import LaunchpadDirectory, _resolve
class FakeResolveFactory:

    def __init__(self, test, expected_path, result):
        self._test = test
        self._expected_path = expected_path
        self._result = result

    def __call__(self, path, url):
        self._test.assertEqual(self._expected_path, path)
        return self._result