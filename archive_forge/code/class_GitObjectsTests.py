import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
class GitObjectsTests(ExternalBase):

    def run_simple(self, format):
        tree = self.make_branch_and_tree('.', format=format)
        self.build_tree(['a/', 'a/foo'])
        tree.add(['a'])
        tree.commit('add a')
        output, error = self.run_bzr('git-objects')
        shas = list(output.splitlines())
        self.assertEqual([40, 40], [len(s) for s in shas])
        self.assertEqual(error, '')
        output, error = self.run_bzr('git-object %s' % shas[0])
        self.assertEqual('', error)

    def test_in_native(self):
        self.run_simple(format='git')

    def test_in_bzr(self):
        self.run_simple(format='2a')