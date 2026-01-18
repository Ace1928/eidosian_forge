from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def RegisterStub(self, service, stub):
    """Register the provided stub for the specified service.

    Args:
      service: string
      stub: stub
    """
    assert service not in self.__stub_map, repr(service)
    self.ReplaceStub(service, stub)