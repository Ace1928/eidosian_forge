from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def __internal_callback(self):
    """This is the callback set on the low-level RPC object.

    It sets a flag on the current object indicating that the high-level
    callback should now be called.  If interrupts are enabled, it also
    interrupts the current wait_any() call by raising an exception.
    """
    self.__must_call_user_callback = True
    self.__rpc.callback = None
    if self.__class__.__local.may_interrupt_wait and (not self.__rpc.exception):
        raise apiproxy_errors.InterruptedError(None, self.__rpc)