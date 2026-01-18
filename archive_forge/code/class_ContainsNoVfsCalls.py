from testtools.matchers import Matcher, Mismatch
from breezy.bzr.smart import vfs
from breezy.bzr.smart.request import request_handlers as smart_request_handlers
class ContainsNoVfsCalls(Matcher):
    """Ensure that none of the specified calls are HPSS calls."""

    def __str__(self):
        return 'ContainsNoVfsCalls()'

    @classmethod
    def match(cls, hpss_calls):
        vfs_calls = []
        for call in hpss_calls:
            try:
                request_method = smart_request_handlers.get(call.call.method)
            except KeyError:
                continue
            if issubclass(request_method, vfs.VfsRequest):
                vfs_calls.append(call.call)
        if len(vfs_calls) == 0:
            return None
        return _NoVfsCallsMismatch(vfs_calls)