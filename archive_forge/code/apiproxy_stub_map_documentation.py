from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
Wait until all given RPCs are finished.

    This is a thin wrapper around wait_any() that loops until all
    given RPCs have finished.

    Args:
      rpcs: Iterable collection of UserRPC instances.

    Returns:
      None.
    