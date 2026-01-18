import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner
class TestShelveUnshelve(TestCaseWithTransport):

    def test_directory(self):
        """Test --directory option"""
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/a', b'initial\n')])
        tree.add('a')
        tree.commit(message='committed')
        self.build_tree_contents([('tree/a', b'initial\nmore\n')])
        self.run_bzr('shelve -d tree --all')
        self.assertFileEqual(b'initial\n', 'tree/a')
        self.run_bzr('unshelve --directory tree')
        self.assertFileEqual(b'initial\nmore\n', 'tree/a')