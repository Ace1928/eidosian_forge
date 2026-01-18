from breezy import branch, errors, uncommit
from breezy.tests import per_branch
class TestUncommitHook(per_branch.TestCaseWithBranch):

    def setUp(self):
        self.hook_calls = []
        super().setUp()

    def capture_post_uncommit_hook(self, local, master, old_revno, old_revid, new_revno, new_revid):
        """Capture post uncommit hook calls to self.hook_calls.

        The call is logged, as is some state of the two branches.
        """
        if local:
            local_locked = local.is_locked()
            local_base = local.base
        else:
            local_locked = None
            local_base = None
        self.hook_calls.append(('post_uncommit', local_base, master.base, old_revno, old_revid, new_revno, new_revid, local_locked, master.is_locked()))

    def test_post_uncommit_to_origin(self):
        tree = self.make_branch_and_memory_tree('branch')
        tree.lock_write()
        tree.add('')
        revid = tree.commit('a revision')
        tree.unlock()
        branch.Branch.hooks.install_named_hook('post_uncommit', self.capture_post_uncommit_hook, None)
        uncommit.uncommit(tree.branch)
        self.assertEqual([('post_uncommit', None, tree.branch.base, 1, revid, 0, None, None, True)], self.hook_calls)

    def test_post_uncommit_bound(self):
        master = self.make_branch('master')
        tree = self.make_branch_and_memory_tree('local')
        try:
            tree.branch.bind(master)
        except branch.BindingUnsupported:
            return
        tree.lock_write()
        tree.add('')
        revid = tree.commit('a revision')
        tree.unlock()
        branch.Branch.hooks.install_named_hook('post_uncommit', self.capture_post_uncommit_hook, None)
        uncommit.uncommit(tree.branch)
        self.assertEqual([('post_uncommit', tree.branch.base, master.base, 1, revid, 0, None, True, True)], self.hook_calls)

    def test_post_uncommit_not_to_origin(self):
        tree = self.make_branch_and_memory_tree('branch')
        tree.lock_write()
        tree.add('')
        revid = tree.commit('first revision')
        revid2 = tree.commit('second revision')
        revid3 = tree.commit('third revision')
        tree.unlock()
        branch.Branch.hooks.install_named_hook('post_uncommit', self.capture_post_uncommit_hook, None)
        uncommit.uncommit(tree.branch, revno=2)
        self.assertEqual([('post_uncommit', None, tree.branch.base, 3, revid3, 1, revid, None, True)], self.hook_calls)