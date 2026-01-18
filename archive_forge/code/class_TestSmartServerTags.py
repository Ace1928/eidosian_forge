from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
class TestSmartServerTags(TestCaseWithTransport):

    def test_set_tag(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/foo', b'thecontents')])
        t.add('foo')
        t.commit('message')
        self.reset_smart_call_log()
        out, err = self.run_bzr(['tag', '-d', self.get_url('branch'), 'tagname'])
        self.assertLength(9, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)

    def test_show_tags(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/foo', b'thecontents')])
        t.add('foo')
        t.commit('message')
        t.branch.tags.set_tag('sometag', b'rev1')
        t.branch.tags.set_tag('sometag', b'rev2')
        self.reset_smart_call_log()
        out, err = self.run_bzr(['tags', '-d', self.get_url('branch')])
        self.assertLength(6, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)