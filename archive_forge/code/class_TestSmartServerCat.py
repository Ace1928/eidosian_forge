from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
class TestSmartServerCat(TestCaseWithTransport):

    def test_simple_branch_cat(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/foo', b'thecontents')])
        t.add('foo')
        t.commit('message')
        self.reset_smart_call_log()
        out, err = self.run_bzr(['cat', '%s/foo' % self.get_url('branch')])
        self.assertLength(9, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)