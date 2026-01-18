from ...revision import Revision
from ...tests import TestCase, TestCaseWithTransport
from .cmds import collapse_by_person, get_revisions_and_committers
class TestGetRevisionsAndCommitters(TestCaseWithTransport):

    def test_simple(self):
        wt = self.make_branch_and_tree('.')
        wt.commit(message='1', committer='Fero <fero@example.com>', rev_id=b'1')
        wt.commit(message='2', committer='Fero <fero@example.com>', rev_id=b'2')
        wt.commit(message='3', committer='Jano <jano@example.com>', rev_id=b'3')
        wt.commit(message='4', committer='Jano <jano@example.com>', authors=['Vinco <vinco@example.com>'], rev_id=b'4')
        wt.commit(message='5', committer='Ferko <fero@example.com>', rev_id=b'5')
        revs, committers = get_revisions_and_committers(wt.branch.repository, [b'1', b'2', b'3', b'4', b'5'])
        fero = ('Fero', 'fero@example.com')
        jano = ('Jano', 'jano@example.com')
        vinco = ('Vinco', 'vinco@example.com')
        ferok = ('Ferko', 'fero@example.com')
        self.assertEqual({fero: fero, jano: jano, vinco: vinco, ferok: fero}, committers)

    def test_empty_email(self):
        wt = self.make_branch_and_tree('.')
        wt.commit(message='1', committer='Fero', rev_id=b'1')
        wt.commit(message='2', committer='Fero', rev_id=b'2')
        wt.commit(message='3', committer='Jano', rev_id=b'3')
        revs, committers = get_revisions_and_committers(wt.branch.repository, [b'1', b'2', b'3'])
        self.assertEqual({('Fero', ''): ('Fero', ''), ('Jano', ''): ('Jano', '')}, committers)

    def test_different_case(self):
        wt = self.make_branch_and_tree('.')
        wt.commit(message='1', committer='Fero', rev_id=b'1')
        wt.commit(message='2', committer='Fero', rev_id=b'2')
        wt.commit(message='3', committer='FERO', rev_id=b'3')
        revs, committers = get_revisions_and_committers(wt.branch.repository, [b'1', b'2', b'3'])
        self.assertEqual({('Fero', ''): ('Fero', ''), ('FERO', ''): ('Fero', '')}, committers)
        self.assertEqual([b'1', b'2', b'3'], sorted([r.revision_id for r in revs]))