from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def ReplaceStub(self, service, stub):
    """Replace the existing stub for the specified service with a new one.

    NOTE: This is a risky operation; external callers should use this with
    caution.

    Args:
      service: string
      stub: stub
    """
    self.__stub_map[service] = stub
    if service == 'datastore':
        self.RegisterStub('datastore_v3', stub)