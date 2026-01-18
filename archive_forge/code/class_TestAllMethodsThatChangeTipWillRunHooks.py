from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
class TestAllMethodsThatChangeTipWillRunHooks(ChangeBranchTipTestCase):
    """Every method of Branch that changes a branch tip will invoke the
    pre/post_change_branch_tip hooks.
    """

    def setUp(self):
        super().setUp()
        self.installPreAndPostHooks()

    def installPreAndPostHooks(self):
        self.pre_hook_calls = self.install_logging_hook('pre')
        self.post_hook_calls = self.install_logging_hook('post')

    def resetHookCalls(self):
        del self.pre_hook_calls[:], self.post_hook_calls[:]

    def assertPreAndPostHooksWereInvoked(self, branch, smart_enabled):
        """assert that both pre and post hooks were called

        :param smart_enabled: The method invoked is one that should be
            smart server ready.
        """
        if smart_enabled and isinstance(branch, remote.RemoteBranch):
            length = 2
        else:
            length = 1
        self.assertEqual(length, len(self.pre_hook_calls))
        self.assertEqual(length, len(self.post_hook_calls))

    def test_set_last_revision_info(self):
        branch = self.make_branch('')
        branch.set_last_revision_info(0, revision.NULL_REVISION)
        self.assertPreAndPostHooksWereInvoked(branch, True)

    def test_generate_revision_history(self):
        branch = self.make_branch('')
        branch.generate_revision_history(revision.NULL_REVISION)
        self.assertPreAndPostHooksWereInvoked(branch, True)

    def test_pull(self):
        source_branch = self.make_branch_with_revision_ids(b'rev-1', b'rev-2')
        self.resetHookCalls()
        destination_branch = self.make_branch('destination')
        destination_branch.pull(source_branch)
        self.assertPreAndPostHooksWereInvoked(destination_branch, False)

    def test_push(self):
        source_branch = self.make_branch_with_revision_ids(b'rev-1', b'rev-2')
        self.resetHookCalls()
        destination_branch = self.make_branch('destination')
        source_branch.push(destination_branch)
        self.assertPreAndPostHooksWereInvoked(destination_branch, True)