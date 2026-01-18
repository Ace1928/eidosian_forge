from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
class TestReplayWorkingtree(TestCaseWithTransport):

    def test_conflicts(self):
        wt = self.make_branch_and_tree('old')
        wt.commit('base', rev_id=b'base')
        self.build_tree(['old/afile'])
        wt.add(['afile'], ids=[b'originalid'])
        wt.commit('bla', rev_id=b'oldparent')
        with open('old/afile', 'w') as f:
            f.write('bloe')
        wt.commit('bla', rev_id=b'oldcommit')
        oldrepos = wt.branch.repository
        wt = self.make_branch_and_tree('new')
        self.build_tree(['new/afile'])
        wt.add(['afile'], ids=[b'newid'])
        wt.commit('bla', rev_id=b'newparent')
        wt.branch.repository.fetch(oldrepos)
        with wt.lock_write():
            replayer = WorkingTreeRevisionRewriter(wt, RebaseState1(wt))
            self.assertRaises(ConflictsInTree, replayer, b'oldcommit', b'newcommit', [b'newparent'])

    def test_simple(self):
        wt = self.make_branch_and_tree('old')
        wt.commit('base', rev_id=b'base')
        self.build_tree(['old/afile'])
        wt.add(['afile'], ids=[b'originalid'])
        wt.commit('bla', rev_id=b'oldparent')
        with open('old/afile', 'w') as f:
            f.write('bloe')
        wt.commit('bla', rev_id=b'oldcommit')
        wt = wt.controldir.sprout('new').open_workingtree()
        self.build_tree(['new/bfile'])
        wt.add(['bfile'], ids=[b'newid'])
        wt.commit('bla', rev_id=b'newparent')
        replayer = WorkingTreeRevisionRewriter(wt, RebaseState1(wt))
        replayer(b'oldcommit', b'newcommit', [b'newparent'])
        oldrev = wt.branch.repository.get_revision(b'oldcommit')
        newrev = wt.branch.repository.get_revision(b'newcommit')
        self.assertEqual([b'newparent'], newrev.parent_ids)
        self.assertEqual(b'newcommit', newrev.revision_id)
        self.assertEqual(oldrev.timestamp, newrev.timestamp)
        self.assertEqual(oldrev.timezone, newrev.timezone)

    def test_multiple(self):
        wt = self.make_branch_and_tree('old')
        wt.commit('base', rev_id=b'base')
        self.build_tree_contents([('old/afile', 'base content')])
        wt.add(['afile'], ids=[b'originalid'])
        wt.commit('bla', rev_id=b'oldparent')
        with open('old/afile', 'w') as f:
            f.write('bloe')
        wt.add_pending_merge(b'ghost')
        wt.commit('bla', rev_id=b'oldcommit')
        new_tree = wt.controldir.sprout('new', revision_id=b'base').open_workingtree()
        new_tree.branch.repository.fetch(wt.branch.repository)
        wt = new_tree
        self.build_tree_contents([('new/afile', 'base content')])
        wt.add(['afile'], ids=[b'originalid'])
        wt.commit('bla', rev_id=b'newparent')
        wt.lock_write()
        replayer = WorkingTreeRevisionRewriter(wt, RebaseState1(wt))
        replayer(b'oldcommit', b'newcommit', (b'newparent', b'ghost'))
        wt.unlock()
        oldrev = wt.branch.repository.get_revision(b'oldcommit')
        newrev = wt.branch.repository.get_revision(b'newcommit')
        self.assertEqual([b'oldparent', b'ghost'], oldrev.parent_ids)
        self.assertEqual([b'newparent', b'ghost'], newrev.parent_ids)
        self.assertEqual(b'newcommit', newrev.revision_id)
        self.assertEqual(oldrev.timestamp, newrev.timestamp)
        self.assertEqual(oldrev.timezone, newrev.timezone)

    def test_already_merged(self):
        """We need to use a merge base that makes sense.

        A
        | \\
        B  D
        | \\|
        C  E

        Rebasing E on C should result in:

        A -> B -> C -> D' -> E'

        Ancestry:
        A:
        B: A
        C: A, B
        D: A
        E: A, B, D
        D': A, B, C
        E': A, B, C, D'

        """
        oldwt = self.make_branch_and_tree('old')
        self.build_tree(['old/afile'])
        with open('old/afile', 'w') as f:
            f.write('A\n' * 10)
        oldwt.add(['afile'])
        oldwt.commit('base', rev_id=b'A')
        newwt = oldwt.controldir.sprout('new').open_workingtree()
        with open('old/afile', 'w') as f:
            f.write('A\n' * 10 + 'B\n')
        oldwt.commit('bla', rev_id=b'B')
        with open('old/afile', 'w') as f:
            f.write('A\n' * 10 + 'C\n')
        oldwt.commit('bla', rev_id=b'C')
        self.build_tree(['new/bfile'])
        newwt.add(['bfile'])
        with open('new/bfile', 'w') as f:
            f.write('D\n')
        newwt.commit('bla', rev_id=b'D')
        with open('new/afile', 'w') as f:
            f.write('E\n' + 'A\n' * 10 + 'B\n')
        with open('new/bfile', 'w') as f:
            f.write('D\nE\n')
        newwt.add_pending_merge(b'B')
        newwt.commit('bla', rev_id=b'E')
        newwt.branch.repository.fetch(oldwt.branch.repository)
        newwt.lock_write()
        replayer = WorkingTreeRevisionRewriter(newwt, RebaseState1(newwt))
        replayer(b'D', b"D'", [b'C'])
        newwt.unlock()
        oldrev = newwt.branch.repository.get_revision(b'D')
        newrev = newwt.branch.repository.get_revision(b"D'")
        self.assertEqual([b'C'], newrev.parent_ids)
        newwt.lock_write()
        replayer = WorkingTreeRevisionRewriter(newwt, RebaseState1(newwt))
        self.assertRaises(ConflictsInTree, replayer, b'E', b"E'", [b"D'"])
        newwt.unlock()
        with open('new/afile') as f:
            self.assertEqual('E\n' + 'A\n' * 10 + 'C\n', f.read())
        newwt.set_conflicts([])
        oldrev = newwt.branch.repository.get_revision(b'E')
        replayer.commit_rebase(oldrev, b"E'")
        newrev = newwt.branch.repository.get_revision(b"E'")
        self.assertEqual([b"D'"], newrev.parent_ids)
        self.assertThat(newwt.branch, RevisionHistoryMatches([b'A', b'B', b'C', b"D'", b"E'"]))