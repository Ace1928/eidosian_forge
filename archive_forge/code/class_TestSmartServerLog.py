from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
class TestSmartServerLog(TestCaseWithTransport):

    def test_standard_log(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/foo', b'thecontents')])
        t.add('foo')
        t.commit('message')
        self.reset_smart_call_log()
        out, err = self.run_bzr(['log', self.get_url('branch')])
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)
        self.assertLength(1, self.hpss_connections)
        self.assertLength(9, self.hpss_calls)

    def test_verbose_log(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/foo', b'thecontents')])
        t.add('foo')
        t.commit('message')
        self.reset_smart_call_log()
        out, err = self.run_bzr(['log', '-v', self.get_url('branch')])
        self.assertLength(10, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)

    def test_per_file(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/foo', b'thecontents')])
        t.add('foo')
        t.commit('message')
        self.reset_smart_call_log()
        out, err = self.run_bzr(['log', '-v', self.get_url('branch') + '/foo'])
        self.assertLength(14, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)