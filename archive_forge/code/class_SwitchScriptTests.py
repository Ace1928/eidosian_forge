import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
class SwitchScriptTests(TestCaseWithTransportAndScript):

    def test_switch_preserves(self):
        self.run_script('\n$ brz init --git r\nCreated a standalone tree (format: git)\n$ cd r\n$ echo original > file.txt\n$ brz add\nadding file.txt\n$ brz ci -q -m "Initial"\n$ echo "entered on master branch" > file.txt\n$ brz stat\nmodified:\n  file.txt\n$ brz switch -b other\n2>Tree is up to date at revision 1.\n2>Switched to branch other\n$ cat file.txt\nentered on master branch\n')