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
class TestBzrDirScenarios(tests.TestCase):

    def test_scenarios(self):
        from .per_controldir import make_scenarios
        vfs_factory = 'v'
        server1 = 'a'
        server2 = 'b'
        formats = ['c', 'd']
        scenarios = make_scenarios(vfs_factory, server1, server2, formats)
        self.assertEqual([('str', {'bzrdir_format': 'c', 'transport_readonly_server': 'b', 'transport_server': 'a', 'vfs_transport_factory': 'v'}), ('str', {'bzrdir_format': 'd', 'transport_readonly_server': 'b', 'transport_server': 'a', 'vfs_transport_factory': 'v'})], scenarios)