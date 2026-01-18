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
class TestSelftestWithIdList(tests.TestCaseInTempDir, SelfTestHelper):

    def test_load_list(self):
        test_id_line = b'%s\n' % self.id().encode('ascii')
        self.build_tree_contents([('test.list', test_id_line)])
        stream = self.run_selftest(load_list='test.list', list_only=True)
        self.assertEqual(test_id_line, stream.getvalue())

    def test_load_unknown(self):
        self.assertRaises(transport.NoSuchFile, self.run_selftest, load_list='missing file name', list_only=True)