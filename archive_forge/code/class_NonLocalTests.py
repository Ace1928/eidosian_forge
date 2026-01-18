import os
import subprocess
import sys
import breezy.branch
import breezy.bzr.branch
from ... import (branch, bzr, config, controldir, errors, help_topics, lock,
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ... import urlutils, win32utils
from ...errors import (NotBranchError, UnknownFormatError,
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from ...transport import memory, pathfilter
from ...transport.http.urllib import HttpTransport
from ...transport.nosmart import NoSmartTransportDecorator
from ...transport.readonly import ReadonlyTransportDecorator
from .. import branch as bzrbranch
from .. import (bzrdir, knitpack_repo, knitrepo, remote, workingtree_3,
from ..fullhistory import BzrBranchFormat5
class NonLocalTests(TestCaseWithTransport):
    """Tests for bzrdir static behaviour on non local paths."""

    def setUp(self):
        super().setUp()
        self.vfs_transport_factory = memory.MemoryServer

    def test_create_branch_convenience(self):
        format = controldir.format_registry.make_controldir('knit')
        branch = bzrdir.BzrDir.create_branch_convenience(self.get_url('foo'), format=format)
        self.assertRaises(errors.NoWorkingTree, branch.controldir.open_workingtree)
        branch.controldir.open_repository()

    def test_create_branch_convenience_force_tree_not_local_fails(self):
        format = controldir.format_registry.make_controldir('knit')
        self.assertRaises(errors.NotLocalUrl, bzrdir.BzrDir.create_branch_convenience, self.get_url('foo'), force_new_tree=True, format=format)
        t = self.get_transport()
        self.assertFalse(t.has('foo'))

    def test_clone(self):
        format = controldir.format_registry.make_controldir('knit')
        branch = bzrdir.BzrDir.create_branch_convenience('local', format=format)
        branch.controldir.open_workingtree()
        result = branch.controldir.clone(self.get_url('remote'))
        self.assertRaises(errors.NoWorkingTree, result.open_workingtree)
        result.open_branch()
        result.open_repository()

    def test_checkout_metadir(self):
        self.make_branch('branch-knit2', format='dirstate-with-subtree')
        my_bzrdir = bzrdir.BzrDir.open(self.get_url('branch-knit2'))
        checkout_format = my_bzrdir.checkout_metadir()
        self.assertIsInstance(checkout_format.workingtree_format, workingtree_4.WorkingTreeFormat4)