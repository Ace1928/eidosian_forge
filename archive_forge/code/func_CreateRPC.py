from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def CreateRPC(service, stubmap=None):
    """Creates a RPC instance for the given service.

  The instance is suitable for talking to remote services.
  Each RPC instance can be used only once, and should not be reused.

  Args:
    service: string representing which service to call.
    stubmap: optional APIProxyStubMap instance, for dependency injection.

  Returns:
    the rpc object.

  Raises:
    AssertionError or RuntimeError if the stub for service doesn't supply a
    CreateRPC method.
  """
    if stubmap is None:
        stubmap = apiproxy
    stub = stubmap.GetStub(service)
    assert stub, 'No api proxy found for service "%s"' % service
    assert hasattr(stub, 'CreateRPC'), ('The service "%s" doesn\'t have ' + 'a CreateRPC method.') % service
    return stub.CreateRPC()