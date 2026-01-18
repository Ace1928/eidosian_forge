from breezy import errors
from breezy.branch import BindingUnsupported, Branch
from breezy.controldir import ControlDir
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable
from breezy.tests.per_interbranch import TestCaseWithInterBranch
class TestPullHook(TestCaseWithInterBranch):

    def setUp(self):
        self.hook_calls = []
        super().setUp()

    def capture_post_pull_hook(self, result):
        """Capture post pull hook calls to self.hook_calls.

        The call is logged, as is some state of the two branches.
        """
        if result.local_branch:
            local_locked = result.local_branch.is_locked()
            local_base = result.local_branch.base
        else:
            local_locked = None
            local_base = None
        self.hook_calls.append(('post_pull', result.source_branch, local_base, result.master_branch.base, result.old_revno, result.old_revid, result.new_revno, result.new_revid, result.source_branch.is_locked(), local_locked, result.master_branch.is_locked()))

    def test_post_pull_empty_history(self):
        target = self.make_to_branch('target')
        source = self.make_from_branch('source')
        Branch.hooks.install_named_hook('post_pull', self.capture_post_pull_hook, None)
        target.pull(source)
        self.assertEqual([('post_pull', source, None, target.base, 0, NULL_REVISION, 0, NULL_REVISION, True, None, True)], self.hook_calls)

    def test_post_pull_bound_branch(self):
        target = self.make_to_branch('target')
        local = self.make_from_branch('local')
        try:
            local.bind(target)
        except BindingUnsupported:
            local = ControlDir.create_branch_convenience('local2')
            local.bind(target)
        source = self.make_from_branch('source')
        Branch.hooks.install_named_hook('post_pull', self.capture_post_pull_hook, None)
        local.pull(source)
        self.assertEqual([('post_pull', source, local.base, target.base, 0, NULL_REVISION, 0, NULL_REVISION, True, True, True)], self.hook_calls)

    def test_post_pull_nonempty_history(self):
        target = self.make_to_branch_and_memory_tree('target')
        target.lock_write()
        target.add('')
        rev1 = target.commit('rev 1')
        target.unlock()
        sourcedir = target.controldir.clone(self.get_url('source'))
        source = sourcedir.open_branch().create_memorytree()
        rev2 = source.commit('rev 2')
        Branch.hooks.install_named_hook('post_pull', self.capture_post_pull_hook, None)
        target.branch.pull(source.branch)
        self.assertEqual([('post_pull', source.branch, None, target.branch.base, 1, rev1, 2, rev2, True, None, True)], self.hook_calls)