from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
class APIProxyStubMap(object):
    """Container of APIProxy stubs for more convenient unittesting.

  Stubs may be either trivial implementations of APIProxy services (e.g.
  DatastoreFileStub, UserServiceStub) or "real" implementations.

  For unittests, we may want to mix and match real and trivial implementations
  of services in order to better focus testing on individual service
  implementations. To achieve this, we allow the client to attach stubs to
  service names, as well as define a default stub to be used if no specific
  matching stub is identified.
  """

    def __init__(self, default_stub=None):
        """Constructor.

    Args:
      default_stub: optional stub

    'default_stub' will be used whenever no specific matching stub is found.
    """
        self.__stub_map = {}
        self.__default_stub = default_stub
        self.__precall_hooks = ListOfHooks()
        self.__postcall_hooks = ListOfHooks()

    def GetPreCallHooks(self):
        """Gets a collection for all precall hooks."""
        return self.__precall_hooks

    def GetPostCallHooks(self):
        """Gets a collection for all precall hooks."""
        return self.__postcall_hooks

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

    def RegisterStub(self, service, stub):
        """Register the provided stub for the specified service.

    Args:
      service: string
      stub: stub
    """
        assert service not in self.__stub_map, repr(service)
        self.ReplaceStub(service, stub)

    def GetStub(self, service):
        """Retrieve the stub registered for the specified service.

    Args:
      service: string

    Returns:
      stub

    Returns the stub registered for 'service', and returns the default stub
    if no such stub is found.
    """
        return self.__stub_map.get(service, self.__default_stub)

    def _CopyStubMap(self):
        """Get a copy of the stub map. For testing only.

    Returns:
      Get a shallow copy of the stub map.
    """
        return dict(self.__stub_map)

    def MakeSyncCall(self, service, call, request, response):
        """The APIProxy entry point.

    Args:
      service: string representing which service to call
      call: string representing which function to call
      request: protocol buffer for the request
      response: protocol buffer for the response

    Returns:
      Response protocol buffer or None. Some implementations may return
      a response protocol buffer instead of modifying 'response'.
      Caller must use returned value in such cases. If 'response' is modified
      then returns None.

    Raises:
      apiproxy_errors.Error or a subclass.
    """
        stub = self.GetStub(service)
        assert stub, 'No api proxy found for service "%s"' % service
        if hasattr(stub, 'CreateRPC'):
            rpc = stub.CreateRPC()
            self.__precall_hooks.Call(service, call, request, response, rpc)
            try:
                rpc.MakeCall(service, call, request, response)
                rpc.Wait()
                rpc.CheckSuccess()
            except Exception as err:
                self.__postcall_hooks.Call(service, call, request, response, rpc, err)
                raise
            else:
                self.__postcall_hooks.Call(service, call, request, response, rpc)
        else:
            self.__precall_hooks.Call(service, call, request, response)
            try:
                returned_response = stub.MakeSyncCall(service, call, request, response)
            except Exception as err:
                self.__postcall_hooks.Call(service, call, request, response, None, err)
                raise
            else:
                self.__postcall_hooks.Call(service, call, request, returned_response or response)
                return returned_response

    def CancelApiCalls(self):
        if self.__default_stub:
            self.__default_stub.CancelApiCalls()