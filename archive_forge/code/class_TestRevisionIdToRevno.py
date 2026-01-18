from breezy import errors
from breezy.tests import TestNotApplicable
from breezy.tests.per_branch import TestCaseWithBranch
class TestRevisionIdToRevno(TestCaseWithBranch):

    def test_simple_revno(self):
        tree, revmap = self.create_tree_with_merge()
        the_branch = tree.branch
        self.assertEqual(0, the_branch.revision_id_to_revno(b'null:'))
        self.assertEqual(1, the_branch.revision_id_to_revno(revmap['1']))
        self.assertEqual(2, the_branch.revision_id_to_revno(revmap['2']))
        self.assertEqual(3, the_branch.revision_id_to_revno(revmap['3']))
        self.assertRaises(errors.NoSuchRevision, the_branch.revision_id_to_revno, b'rev-none')
        self.assertRaises(errors.NoSuchRevision, the_branch.revision_id_to_revno, revmap['1.1.1'])

    def test_mainline_ghost(self):
        tree = self.make_branch_and_tree('tree1')
        if not tree.branch.repository._format.supports_ghosts:
            raise TestNotApplicable('repository format does not support ghosts')
        tree.set_parent_ids([b'spooky'], allow_leftmost_as_ghost=True)
        tree.add('')
        tree.commit('msg1', rev_id=b'rev1')
        tree.commit('msg2', rev_id=b'rev2')
        self.assertRaises((errors.NoSuchRevision, errors.GhostRevisionsHaveNoRevno), tree.branch.revision_id_to_revno, b'unknown')
        self.assertEqual(1, tree.branch.revision_id_to_revno(b'rev1'))
        self.assertEqual(2, tree.branch.revision_id_to_revno(b'rev2'))

    def test_empty(self):
        branch = self.make_branch('.')
        self.assertRaises(errors.NoSuchRevision, branch.revision_id_to_revno, b'unknown')
        self.assertEqual(0, branch.revision_id_to_revno(b'null:'))