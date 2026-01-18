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
class TestConvenienceMakers(tests.TestCaseWithTransport):
    """Test for the make_* convenience functions."""

    def test_make_branch_and_tree_with_format(self):
        self.make_branch_and_tree('a', format=breezy.bzr.bzrdir.BzrDirMetaFormat1())
        self.assertIsInstance(breezy.controldir.ControlDir.open('a')._format, breezy.bzr.bzrdir.BzrDirMetaFormat1)

    def test_make_branch_and_memory_tree(self):
        tree = self.make_branch_and_memory_tree('a')
        self.assertIsInstance(tree, breezy.memorytree.MemoryTree)

    def test_make_tree_for_local_vfs_backed_transport(self):
        self.transport_server = test_server.FakeVFATServer
        self.assertFalse(self.get_url('t1').startswith('file://'))
        tree = self.make_branch_and_tree('t1')
        base = tree.controldir.root_transport.base
        self.assertStartsWith(base, 'file://')
        self.assertEqual(tree.controldir.root_transport, tree.branch.controldir.root_transport)
        self.assertEqual(tree.controldir.root_transport, tree.branch.repository.controldir.root_transport)