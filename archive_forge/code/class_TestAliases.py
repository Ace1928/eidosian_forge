import os
from breezy import config
from breezy.branch import Branch
from breezy.tests import TestCaseWithTransport
from breezy.trace import mutter
class TestAliases(TestCaseWithTransport):

    def test_aliases(self):

        def bzr(args, **kwargs):
            return self.run_bzr(args, **kwargs)[0]

        def bzr_catch_error(args, **kwargs):
            return self.run_bzr(args, **kwargs)[1]
        conf = config.GlobalConfig.from_string(b'[ALIASES]\nc=cat\nc1=cat -r 1\nc2=cat -r 1 -r2\n', save=True)
        str1 = 'foo\n'
        str2 = 'bar\n'
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('a', str1)])
        tree.add('a')
        tree.commit(message='1')
        self.assertEqual(bzr('c a'), str1)
        self.build_tree_contents([('a', str2)])
        tree.commit(message='2')
        self.assertEqual(bzr('c a'), str2)
        self.assertEqual(bzr('c1 a'), str1)
        self.assertEqual(bzr('c1 --revision 2 a'), str2)
        bzr('--no-aliases c a', retcode=3)
        self.assertEqual(bzr_catch_error('--no-aliases c a', retcode=None), 'brz: ERROR: unknown command "c". Perhaps you meant "ci"\n')
        bzr('c -r1 -r2', retcode=3)
        bzr('c1 -r1 -r2', retcode=3)
        bzr('c2', retcode=3)
        bzr('c2 -r1', retcode=3)