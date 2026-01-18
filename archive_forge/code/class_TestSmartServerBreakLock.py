from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
class TestSmartServerBreakLock(TestCaseWithTransport):

    def test_simple_branch_break_lock(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('branch')
        t.branch.lock_write()
        self.reset_smart_call_log()
        out, err = self.run_bzr(['break-lock', '--force', self.get_url('branch')])
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)
        self.assertLength(1, self.hpss_connections)
        self.assertLength(5, self.hpss_calls)