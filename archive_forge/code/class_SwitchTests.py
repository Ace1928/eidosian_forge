import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
class SwitchTests(ExternalBase):

    def test_switch_branch(self):
        repo = GitRepo.init(self.test_dir)
        builder = tests.GitBranchBuilder()
        builder.set_branch(b'refs/heads/oldbranch')
        builder.set_file('a', b'text for a\n', False)
        builder.commit(b'Joe Foo <joe@foo.com>', '<The commit message>')
        builder.set_branch(b'refs/heads/newbranch')
        builder.reset()
        builder.set_file('a', b'text for new a\n', False)
        builder.commit(b'Joe Foo <joe@foo.com>', '<The commit message>')
        builder.finish()
        repo.refs.set_symbolic_ref(b'HEAD', b'refs/heads/newbranch')
        repo.reset_index()
        output, error = self.run_bzr('switch oldbranch')
        self.assertEqual(output, '')
        self.assertTrue(error.startswith('Updated to revision 1.\n'), error)
        self.assertFileEqual('text for a\n', 'a')
        tree = WorkingTree.open('.')
        with tree.lock_read():
            basis_tree = tree.basis_tree()
            with basis_tree.lock_read():
                self.assertEqual([], list(tree.iter_changes(basis_tree)))

    def test_branch_with_nested_trees(self):
        orig = self.make_branch_and_tree('source', format='git')
        subtree = self.make_branch_and_tree('source/subtree', format='git')
        self.build_tree(['source/subtree/a'])
        self.build_tree_contents([('source/.gitmodules', '[submodule "subtree"]\n    path = subtree\n    url = %s\n' % subtree.user_url)])
        subtree.add(['a'])
        subtree.commit('add subtree contents')
        orig.add_reference(subtree)
        orig.add(['.gitmodules'])
        orig.commit('add subtree')
        self.run_bzr('branch source target')
        target = WorkingTree.open('target')
        target_subtree = WorkingTree.open('target/subtree')
        self.assertTreesEqual(orig, target)
        self.assertTreesEqual(subtree, target_subtree)