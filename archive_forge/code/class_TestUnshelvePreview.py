import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner
class TestUnshelvePreview(TestCaseWithTransport):

    def test_non_ascii(self):
        """Test that we can show a non-ascii diff that would result from unshelving"""
        init_content = 'Initial: Изнач\n'.encode()
        more_content = 'More: Ещё\n'.encode()
        next_content = init_content + more_content
        diff_part = b'@@ -1,1 +1,2 @@\n %s+%s' % (init_content, more_content)
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('a_file', init_content)])
        tree.add('a_file')
        tree.commit(message='committed')
        self.build_tree_contents([('a_file', next_content)])
        self.run_bzr(['shelve', '--all'])
        out, err = self.run_bzr_raw(['unshelve', '--preview'], encoding='latin-1')
        self.assertContainsString(out, diff_part)