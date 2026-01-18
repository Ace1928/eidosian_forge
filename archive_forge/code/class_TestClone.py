import os
from breezy import branch, controldir, tests
from breezy.urlutils import local_path_to_url
class TestClone(tests.TestCaseWithTransport):

    def example_dir(self, path='.', format=None):
        tree = self.make_branch_and_tree(path, format=format)
        self.build_tree_contents([(path + '/hello', b'foo')])
        tree.add('hello')
        tree.commit(message='setup')
        self.build_tree_contents([(path + '/goodbye', b'baz')])
        tree.add('goodbye')
        tree.commit(message='setup')
        return tree

    def test_clone(self):
        """Branch from one branch to another."""
        self.example_dir('a')
        self.run_bzr('clone a b')
        b = branch.Branch.open('b')
        self.run_bzr('clone a c -r 1')
        self.assertFalse(b._transport.has('branch-name'))
        b.controldir.open_workingtree().commit(message='foo', allow_pointless=True)

    def test_clone_no_to_location(self):
        """The to_location is derived from the source branch name."""
        os.mkdir('something')
        a = self.example_dir('something/a').branch
        self.run_bzr('clone something/a')
        b = branch.Branch.open('a')
        self.assertEqual(b.last_revision_info(), a.last_revision_info())

    def test_from_colocated(self):
        """Branch from a colocated branch into a regular branch."""
        os.mkdir('b')
        tree = self.example_dir('b/a')
        tree.controldir.create_branch(name='somecolo')
        out, err = self.run_bzr('clone %s' % local_path_to_url('b/a'))
        self.assertEqual('', out)
        self.assertEqual('Created new control directory.\n', err)
        self.assertPathExists('a')
        target = controldir.ControlDir.open('a')
        self.assertEqual(['', 'somecolo'], target.branch_names())