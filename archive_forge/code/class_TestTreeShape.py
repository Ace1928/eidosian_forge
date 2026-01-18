import os
from breezy import tests
from breezy.tests import features
class TestTreeShape(tests.TestCaseWithTransport):

    def test_build_tree(self):
        """Test tree-building test helper"""
        self.build_tree_contents([('foo', b'new contents'), ('.bzr/',), ('.bzr/README', b'hello')])
        self.assertPathExists('foo')
        self.assertPathExists('.bzr/README')
        self.assertFileEqual(b'hello', '.bzr/README')

    def test_build_tree_symlink(self):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        self.build_tree_contents([('link@', 'target')])
        self.assertEqual('target', os.readlink('link'))