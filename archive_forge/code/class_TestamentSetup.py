import os
from breezy import osutils
from breezy.bzr.testament import StrictTestament, StrictTestament3, Testament
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import SymlinkFeature
class TestamentSetup(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.wt = self.make_branch_and_tree('.', format='development-subtree')
        self.wt.set_root_id(b'TREE_ROT')
        b = self.b = self.wt.branch
        b.nick = 'test branch'
        self.wt.commit(message='initial null commit', committer='test@user', timestamp=1129025423, timezone=0, rev_id=b'test@user-1')
        self.build_tree_contents([('hello', b'contents of hello file'), ('src/',), ('src/foo.c', b'int main()\n{\n}\n')])
        self.wt.add(['hello', 'src', 'src/foo.c'], ids=[b'hello-id', b'src-id', b'foo.c-id'])
        tt = self.wt.transform()
        trans_id = tt.trans_id_tree_path('hello')
        tt.set_executability(True, trans_id)
        tt.apply()
        self.wt.commit(message='add files and directories', timestamp=1129025483, timezone=36000, rev_id=b'test@user-2', committer='test@user')