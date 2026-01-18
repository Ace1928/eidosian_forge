from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
@classmethod
def __check_one(cls, rpcs):
    """Check the list of RPCs for one that is finished, or one that is running.

    Args:
      rpcs: Iterable collection of UserRPC instances.

    Returns:
      A pair (finished, running), as follows:
      (UserRPC, None) indicating the first RPC found that is finished;
      (None, UserRPC) indicating the first RPC found that is running;
      (None, None) indicating no RPCs are finished or running.
    """
    rpc = None
    for rpc in rpcs:
        assert isinstance(rpc, cls), repr(rpc)
        state = rpc.__rpc.state
        if state == apiproxy_rpc.RPC.FINISHING:
            rpc.__call_user_callback()
            return (rpc, None)
        assert state != apiproxy_rpc.RPC.IDLE, repr(rpc)
    return (None, rpc)