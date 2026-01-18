from testtools.matchers import Matcher, Mismatch
from breezy.bzr.smart import vfs
from breezy.bzr.smart.request import request_handlers as smart_request_handlers
class _NoVfsCallsMismatch(Mismatch):
    """Mismatch describing a list of HPSS calls which includes VFS requests."""

    def __init__(self, vfs_calls):
        self.vfs_calls = vfs_calls

    def describe(self):
        return 'no VFS calls expected, got: %s' % ','.join(['{}({})'.format(c.method, ', '.join([repr(a) for a in c.args])) for c in self.vfs_calls])