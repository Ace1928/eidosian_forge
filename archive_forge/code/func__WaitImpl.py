from __future__ import absolute_import
import sys
from googlecloudsdk.third_party.appengine._internal import six_subset
def _WaitImpl(self):
    """Override this method to implement a real asynchronous call rpc.

    Returns:
      True if the async call was completed successfully.
    """
    try:
        try:
            self.stub.MakeSyncCall(self.package, self.call, self.request, self.response)
        except Exception:
            _, self._exception, self._traceback = sys.exc_info()
    finally:
        self._state = RPC.FINISHING
        self._Callback()
    return True