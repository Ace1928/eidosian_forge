from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
class TestSmartServerVerifySignatures(TestCaseWithTransport):

    def monkey_patch_gpg(self):
        """Monkey patch the gpg signing strategy to be a loopback.

        This also registers the cleanup, so that we will revert to
        the original gpg strategy when done.
        """
        self.overrideAttr(gpg, 'GPGStrategy', gpg.LoopbackGPGStrategy)

    def test_verify_signatures(self):
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/foo', b'thecontents')])
        t.add('foo')
        t.commit('message')
        self.monkey_patch_gpg()
        out, err = self.run_bzr(['sign-my-commits', self.get_url('branch')])
        self.reset_smart_call_log()
        self.run_bzr('sign-my-commits')
        out = self.run_bzr(['verify-signatures', self.get_url('branch')])
        self.assertLength(10, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertThat(self.hpss_calls, ContainsNoVfsCalls)