from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
class TestStackingConnections(transport_util.TestCaseWithConnectionHookedTransport):

    def setUp(self):
        super().setUp()
        try:
            base_tree = self.make_branch_and_tree('base', format=self.bzrdir_format)
        except errors.UninitializableFormat as e:
            raise TestNotApplicable(e)
        stacked = self.make_branch('stacked', format=self.bzrdir_format)
        try:
            stacked.set_stacked_on_url(base_tree.branch.base)
        except unstackable_format_errors as e:
            raise TestNotApplicable(e)
        self.rev_base = base_tree.commit('first')
        stacked.set_last_revision_info(1, self.rev_base)
        stacked_relative = self.make_branch('stacked_relative', format=self.bzrdir_format)
        stacked_relative.set_stacked_on_url(base_tree.branch.user_url)
        stacked.set_last_revision_info(1, self.rev_base)
        self.start_logging_connections()

    def test_open_stacked(self):
        b = _mod_branch.Branch.open(self.get_url('stacked'))
        rev = b.repository.get_revision(self.rev_base)
        self.assertEqual(1, len(self.connections))

    def test_open_stacked_relative(self):
        b = _mod_branch.Branch.open(self.get_url('stacked_relative'))
        rev = b.repository.get_revision(self.rev_base)
        self.assertEqual(1, len(self.connections))