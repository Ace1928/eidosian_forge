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
class TestMeta1DirColoFormat(TestCaseWithTransport):
    """Tests specific to the meta1 dir with colocated branches format."""

    def test_supports_colo(self):
        format = bzrdir.BzrDirMetaFormat1Colo()
        self.assertTrue(format.colocated_branches)

    def test_upgrade_from_2a(self):
        tree = self.make_branch_and_tree('.', format='2a')
        format = bzrdir.BzrDirMetaFormat1Colo()
        self.assertTrue(tree.controldir.needs_format_conversion(format))
        converter = tree.controldir._format.get_converter(format)
        result = converter.convert(tree.controldir, None)
        self.assertIsInstance(result._format, bzrdir.BzrDirMetaFormat1Colo)
        self.assertFalse(result.needs_format_conversion(format))

    def test_downgrade_to_2a(self):
        tree = self.make_branch_and_tree('.', format='development-colo')
        format = bzrdir.BzrDirMetaFormat1()
        self.assertTrue(tree.controldir.needs_format_conversion(format))
        converter = tree.controldir._format.get_converter(format)
        result = converter.convert(tree.controldir, None)
        self.assertIsInstance(result._format, bzrdir.BzrDirMetaFormat1)
        self.assertFalse(result.needs_format_conversion(format))

    def test_downgrade_to_2a_too_many_branches(self):
        tree = self.make_branch_and_tree('.', format='development-colo')
        tree.controldir.create_branch(name='another-colocated-branch')
        converter = tree.controldir._format.get_converter(bzrdir.BzrDirMetaFormat1())
        result = converter.convert(tree.controldir, bzrdir.BzrDirMetaFormat1())
        self.assertIsInstance(result._format, bzrdir.BzrDirMetaFormat1)

    def test_nested(self):
        tree = self.make_branch_and_tree('.', format='development-colo')
        tree.controldir.create_branch(name='foo')
        tree.controldir.create_branch(name='fool/bla')
        self.assertRaises(errors.ParentBranchExists, tree.controldir.create_branch, name='foo/bar')

    def test_parent(self):
        tree = self.make_branch_and_tree('.', format='development-colo')
        tree.controldir.create_branch(name='fool/bla')
        tree.controldir.create_branch(name='foo/bar')
        self.assertRaises(errors.AlreadyBranchError, tree.controldir.create_branch, name='foo')

    def test_supports_relative_reference(self):
        tree = self.make_branch_and_tree('.', format='development-colo')
        target1 = tree.controldir.create_branch(name='target1')
        target2 = tree.controldir.create_branch(name='target2')
        source = tree.controldir.set_branch_reference(target1, name='source')
        self.assertEqual(target1.user_url, tree.controldir.open_branch('source').user_url)
        source.controldir.get_branch_transport(None, 'source').put_bytes('location', b'file:,branch=target2')
        self.assertEqual(target2.user_url, tree.controldir.open_branch('source').user_url)