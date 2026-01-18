from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
class RebaseStateTests(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.wt = self.make_branch_and_tree('.')
        self.state = RebaseState1(self.wt)

    def test_rebase_plan_exists_false(self):
        self.assertFalse(self.state.has_plan())

    def test_rebase_plan_exists_empty(self):
        self.wt._transport.put_bytes(REBASE_PLAN_FILENAME, b'')
        self.assertFalse(self.state.has_plan())

    def test_rebase_plan_exists(self):
        self.wt._transport.put_bytes(REBASE_PLAN_FILENAME, b'foo')
        self.assertTrue(self.state.has_plan())

    def test_remove_rebase_plan(self):
        self.wt._transport.put_bytes(REBASE_PLAN_FILENAME, b'foo')
        self.state.remove_plan()
        self.assertFalse(self.state.has_plan())

    def test_remove_rebase_plan_twice(self):
        self.state.remove_plan()
        self.assertFalse(self.state.has_plan())

    def test_write_rebase_plan(self):
        with open('hello', 'w') as f:
            f.write('hello world')
        self.wt.add('hello')
        self.wt.commit(message='add hello', rev_id=b'bla')
        self.state.write_plan({b'oldrev': (b'newrev', [b'newparent1', b'newparent2'])})
        self.assertEqualDiff(b'# Bazaar rebase plan 1\n1 bla\noldrev newrev newparent1 newparent2\n', self.wt._transport.get_bytes(REBASE_PLAN_FILENAME))

    def test_read_rebase_plan_nonexistant(self):
        self.assertRaises(NoSuchFile, self.state.read_plan)

    def test_read_rebase_plan_empty(self):
        self.wt._transport.put_bytes(REBASE_PLAN_FILENAME, b'')
        self.assertRaises(NoSuchFile, self.state.read_plan)

    def test_read_rebase_plan(self):
        self.wt._transport.put_bytes(REBASE_PLAN_FILENAME, b'# Bazaar rebase plan 1\n1 bla\noldrev newrev newparent1 newparent2\n')
        self.assertEqual(((1, b'bla'), {b'oldrev': (b'newrev', (b'newparent1', b'newparent2'))}), self.state.read_plan())

    def test_read_nonexistant(self):
        self.assertIs(None, self.state.read_active_revid())

    def test_read_null(self):
        self.wt._transport.put_bytes(REBASE_CURRENT_REVID_FILENAME, NULL_REVISION)
        self.assertIs(None, self.state.read_active_revid())

    def test_read(self):
        self.wt._transport.put_bytes(REBASE_CURRENT_REVID_FILENAME, b'bla')
        self.assertEqual(b'bla', self.state.read_active_revid())

    def test_write(self):
        self.state.write_active_revid(b'bloe')
        self.assertEqual(b'bloe', self.state.read_active_revid())

    def test_write_null(self):
        self.state.write_active_revid(None)
        self.assertIs(None, self.state.read_active_revid())