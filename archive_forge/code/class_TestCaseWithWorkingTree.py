from breezy import branchbuilder, tests, transport, workingtree
from breezy.tests import per_controldir, test_server
from breezy.transport import memory
class TestCaseWithWorkingTree(per_controldir.TestCaseWithControlDir):

    def make_branch_and_tree(self, relpath, format=None):
        made_control = self.make_controldir(relpath, format=format)
        made_control.create_repository()
        b = made_control.create_branch()
        if getattr(self, 'repo_is_remote', False):
            t = transport.get_transport(relpath)
            t.ensure_base()
            bzrdir_format = self.workingtree_format.get_controldir_for_branch()
            wt_dir = bzrdir_format.initialize_on_transport(t)
            branch_ref = wt_dir.set_branch_reference(b)
            wt = wt_dir.create_workingtree(None, from_branch=branch_ref)
        else:
            wt = self.workingtree_format.initialize(made_control)
        return wt

    def make_branch_builder(self, relpath, format=None):
        if format is None:
            format = self.workingtree_format.get_controldir_for_branch()
        builder = branchbuilder.BranchBuilder(self.get_transport(relpath), format=format)
        return builder