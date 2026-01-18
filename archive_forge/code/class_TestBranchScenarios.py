import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
class TestBranchScenarios(tests.TestCase):

    def test_scenarios(self):
        from .per_branch import make_scenarios
        server1 = 'a'
        server2 = 'b'
        formats = [('c', 'C'), ('d', 'D')]
        scenarios = make_scenarios(server1, server2, formats)
        self.assertEqual(2, len(scenarios))
        self.assertEqual([('str', {'branch_format': 'c', 'bzrdir_format': 'C', 'transport_readonly_server': 'b', 'transport_server': 'a'}), ('str', {'branch_format': 'd', 'bzrdir_format': 'D', 'transport_readonly_server': 'b', 'transport_server': 'a'})], scenarios)