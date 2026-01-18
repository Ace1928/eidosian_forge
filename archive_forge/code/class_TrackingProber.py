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
class TrackingProber(BzrProber):
    seen_urls = []

    @classmethod
    def probe_transport(klass, transport):
        klass.seen_urls.append(transport.base)
        return BzrProber.probe_transport(transport)