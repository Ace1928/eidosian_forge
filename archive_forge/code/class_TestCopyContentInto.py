from breezy import branch
from breezy.errors import NoRoundtrippingSupport
from breezy.tests import TestNotApplicable
from breezy.tests.per_interbranch import (StubMatchingInter, StubWithFormat,
class TestCopyContentInto(TestCaseWithInterBranch):

    def test_contract_convenience_method(self):
        self.tree1 = self.make_from_branch_and_tree('tree1')
        rev1 = self.tree1.commit('one')
        branch2 = self.make_to_branch('tree2')
        try:
            branch2.repository.fetch(self.tree1.branch.repository)
        except NoRoundtrippingSupport:
            raise TestNotApplicable('lossless cross-vcs fetch from %r to %r unsupported' % (self.tree1.branch, branch2))
        self.tree1.branch.copy_content_into(branch2, revision_id=rev1)

    def test_inter_is_used(self):
        self.tree1 = self.make_from_branch_and_tree('tree1')
        self.addCleanup(branch.InterBranch.unregister_optimiser, StubMatchingInter)
        branch.InterBranch.register_optimiser(StubMatchingInter)
        del StubMatchingInter._uses[:]
        self.tree1.branch.copy_content_into(StubWithFormat(), revision_id=b'54')
        self.assertLength(1, StubMatchingInter._uses)
        use = StubMatchingInter._uses[0]
        self.assertEqual('copy_content_into', use[1])
        self.assertEqual(b'54', use[3]['revision_id'])
        del StubMatchingInter._uses[:]