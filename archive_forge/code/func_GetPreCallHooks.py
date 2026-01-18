from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def GetPreCallHooks(self):
    """Gets a collection for all precall hooks."""
    return self.__precall_hooks