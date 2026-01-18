from typing import List
from .. import urlutils
from ..branch import Branch
from ..bzr import BzrProber
from ..bzr.branch import BranchReferenceFormat
from ..controldir import ControlDir, ControlDirFormat
from ..errors import NotBranchError, RedirectRequested
from ..transport import (Transport, chroot, get_transport, register_transport,
from ..url_policy_open import (BadUrl, BranchLoopError, BranchOpener,
from . import TestCase, TestCaseWithTransport
class TestRedirects(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        BranchOpener.install_hook()

    def setup_redirect(self, target_url):

        class RedirectingTransport(Transport):

            def get(self, name):
                raise RedirectRequested(self.base, target_url)

            def _redirected_to(self, source, target):
                return get_transport(target)
        register_transport_proto('redirecting://', help='Test transport that redirects.')
        register_transport('redirecting://', RedirectingTransport)
        self.addCleanup(unregister_transport, 'redirecting://', RedirectingTransport)

    def make_branch_opener(self, allowed_urls, probers=None):
        policy = WhitelistPolicy(True, allowed_urls, True)
        return BranchOpener(policy, probers)

    def test_redirect_forbidden(self):
        b = self.make_branch('b')
        self.setup_redirect(b.base)

        class TrackingProber(BzrProber):
            seen_urls = []

            @classmethod
            def probe_transport(klass, transport):
                klass.seen_urls.append(transport.base)
                return BzrProber.probe_transport(transport)
        opener = self.make_branch_opener(['redirecting:///'], probers=[TrackingProber])
        self.assertRaises(BadUrl, opener.open, 'redirecting:///')
        opener = self.make_branch_opener(['redirecting:///', b.base], probers=[TrackingProber])
        opener.open('redirecting:///')