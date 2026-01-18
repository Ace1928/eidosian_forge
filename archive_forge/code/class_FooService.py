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
class FooService:
    """A directory service that maps the name to a FILE url"""

    def look_up(self, name, url, purpose=None):
        if 'lp:///apt' == url:
            return target_branch.base.rstrip('/')
        return '!unexpected look_up value!'