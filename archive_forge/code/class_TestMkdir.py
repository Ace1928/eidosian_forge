import os
from breezy import tests
class TestMkdir(tests.TestCaseWithTransport):

    def test_mkdir(self):
        tree = self.make_branch_and_tree('.')
        self.run_bzr(['mkdir', 'somedir'])
        self.assertEqual(tree.kind('somedir'), 'directory')

    def test_mkdir_parents(self):
        tree = self.make_branch_and_tree('.')
        self.run_bzr(['mkdir', '-p', 'somedir/foo'])
        self.assertEqual(tree.kind('somedir/foo'), 'directory')

    def test_mkdir_parents_existing_versioned_dir(self):
        tree = self.make_branch_and_tree('.')
        tree.mkdir('somedir')
        self.assertEqual(tree.kind('somedir'), 'directory')
        self.run_bzr(['mkdir', '-p', 'somedir'])

    def test_mkdir_parents_existing_unversioned_dir(self):
        tree = self.make_branch_and_tree('.')
        os.mkdir('somedir')
        self.run_bzr(['mkdir', '-p', 'somedir'])
        self.assertEqual(tree.kind('somedir'), 'directory')

    def test_mkdir_parents_with_unversioned_parent(self):
        tree = self.make_branch_and_tree('.')
        os.mkdir('somedir')
        self.run_bzr(['mkdir', '-p', 'somedir/foo'])
        self.assertEqual(tree.kind('somedir/foo'), 'directory')