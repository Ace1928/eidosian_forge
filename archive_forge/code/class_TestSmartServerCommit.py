from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
class TestSmartServerCommit(TestCaseWithTransport):

    def test_commit_to_lightweight(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('from')
        for count in range(9):
            t.commit(message='commit %d' % count)
        out, err = self.run_bzr(['checkout', '--lightweight', self.get_url('from'), 'target'])
        self.reset_smart_call_log()
        self.build_tree(['target/afile'])
        self.run_bzr(['add', 'target/afile'])
        out, err = self.run_bzr(['commit', '-m', 'do something', 'target'])
        self.assertLength(211, self.hpss_calls)
        self.assertLength(2, self.hpss_connections)
        self.expectFailure('commit still uses VFS calls', self.assertThat, self.hpss_calls, ContainsNoVfsCalls)