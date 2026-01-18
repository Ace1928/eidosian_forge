import os
from ... import tests
from ..features import HardlinkFeature
class TestLinkTreeCommand(tests.TestCaseWithTransport):

    def setUp(self):
        tests.TestCaseWithTransport.setUp(self)
        self.requireFeature(HardlinkFeature(self.test_dir))
        self.parent_tree = self.make_branch_and_tree('parent')
        self.parent_tree.lock_write()
        self.addCleanup(self.parent_tree.unlock)
        self.build_tree_contents([('parent/foo', b'bar')])
        self.parent_tree.add('foo', ids=b'foo-id')
        self.parent_tree.commit('added foo')
        child_controldir = self.parent_tree.controldir.sprout('child')
        self.child_tree = child_controldir.open_workingtree()

    def hardlinked(self):
        parent_stat = os.lstat(self.parent_tree.abspath('foo'))
        child_stat = os.lstat(self.child_tree.abspath('foo'))
        return parent_stat.st_ino == child_stat.st_ino

    def test_link_tree(self):
        """Ensure the command works as intended"""
        os.chdir('child')
        self.parent_tree.unlock()
        self.run_bzr('link-tree ../parent')
        self.assertTrue(self.hardlinked())
        self.parent_tree.lock_write()