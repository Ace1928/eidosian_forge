import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
class GitApplyTests(ExternalBase):

    def test_apply(self):
        b = self.make_branch_and_tree('.')
        with open('foo.patch', 'w') as f:
            f.write('From bdefb25fab801e6af0a70e965f60cb48f2b759fa Mon Sep 17 00:00:00 2001\nFrom: Dmitry Bogatov <KAction@debian.org>\nDate: Fri, 8 Feb 2019 23:28:30 +0000\nSubject: [PATCH] Add fixed for out-of-date-standards-version\n\n---\n message           | 3 +++\n 1 files changed, 14 insertions(+)\n create mode 100644 message\n\ndiff --git a/message b/message\nnew file mode 100644\nindex 0000000..05ec0b1\n--- /dev/null\n+++ b/message\n@@ -0,0 +1,3 @@\n+Update standards version, no changes needed.\n+Certainty: certain\n+Fixed-Lintian-Tags: out-of-date-standards-version\n')
        output, error = self.run_bzr('git-apply foo.patch')
        self.assertContainsRe(error, 'Committing to: .*\nCommitted revision 1.\n')