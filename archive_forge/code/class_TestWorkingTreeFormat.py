import os
from io import BytesIO
from .. import (conflicts, errors, symbol_versioning, trace, transport,
from ..bzr import bzrdir
from ..bzr import conflicts as _mod_bzr_conflicts
from ..bzr import workingtree as bzrworkingtree
from ..bzr import workingtree_3, workingtree_4
from ..lock import write_locked
from ..lockdir import LockDir
from ..tree import TreeDirectory, TreeEntry, TreeFile, TreeLink
from . import TestCase, TestCaseWithTransport, TestSkipped
from .features import SymlinkFeature
class TestWorkingTreeFormat(TestCaseWithTransport):
    """Tests for the WorkingTreeFormat facility."""

    def test_find_format_string(self):
        branch = self.make_branch('branch')
        self.assertRaises(errors.NoWorkingTree, bzrworkingtree.WorkingTreeFormatMetaDir.find_format_string, branch.controldir)
        transport = branch.controldir.get_workingtree_transport(None)
        transport.mkdir('.')
        transport.put_bytes('format', b'some format name')
        self.assertEqual(b'some format name', bzrworkingtree.WorkingTreeFormatMetaDir.find_format_string(branch.controldir))

    def test_find_format(self):
        self.build_tree(['foo/', 'bar/'])

        def check_format(format, url):
            dir = format._matchingcontroldir.initialize(url)
            dir.create_repository()
            dir.create_branch()
            format.initialize(dir)
            found_format = bzrworkingtree.WorkingTreeFormatMetaDir.find_format(dir)
            self.assertIsInstance(found_format, format.__class__)
        check_format(workingtree_3.WorkingTreeFormat3(), 'bar')

    def test_find_format_no_tree(self):
        dir = bzrdir.BzrDirMetaFormat1().initialize('.')
        self.assertRaises(errors.NoWorkingTree, bzrworkingtree.WorkingTreeFormatMetaDir.find_format, dir)

    def test_find_format_unknown_format(self):
        dir = bzrdir.BzrDirMetaFormat1().initialize('.')
        dir.create_repository()
        dir.create_branch()
        SampleTreeFormat().initialize(dir)
        self.assertRaises(errors.UnknownFormatError, bzrworkingtree.WorkingTreeFormatMetaDir.find_format, dir)

    def test_find_format_with_features(self):
        tree = self.make_branch_and_tree('.', format='2a')
        tree.update_feature_flags({b'name': b'necessity'})
        found_format = bzrworkingtree.WorkingTreeFormatMetaDir.find_format(tree.controldir)
        self.assertIsInstance(found_format, workingtree.WorkingTreeFormat)
        self.assertEqual(found_format.features.get(b'name'), b'necessity')
        self.assertRaises(bzrdir.MissingFeature, found_format.check_support_status, True)
        self.addCleanup(bzrworkingtree.WorkingTreeFormatMetaDir.unregister_feature, b'name')
        bzrworkingtree.WorkingTreeFormatMetaDir.register_feature(b'name')
        found_format.check_support_status(True)