from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
class TestSmartServerUncommit(TestCaseWithTransport):

    def test_uncommit(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('from')
        for count in range(2):
            t.commit(message='commit %d' % count)
        self.reset_smart_call_log()
        out, err = self.run_bzr(['uncommit', '--force', self.get_url('from')])
        self.assertLength(14, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)