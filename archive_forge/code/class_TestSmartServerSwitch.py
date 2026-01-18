from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
class TestSmartServerSwitch(TestCaseWithTransport):

    def test_switch_lightweight(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('from')
        for count in range(9):
            t.commit(message='commit %d' % count)
        out, err = self.run_bzr(['checkout', '--lightweight', self.get_url('from'), 'target'])
        self.reset_smart_call_log()
        self.run_bzr(['switch', self.get_url('from')], working_dir='target')
        self.assertLength(21, self.hpss_calls)
        self.assertLength(3, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)