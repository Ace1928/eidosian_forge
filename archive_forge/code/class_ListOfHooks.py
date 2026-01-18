from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
class ListOfHooks(object):
    """An ordered collection of hooks for a particular API call.

  A hook is a function that has exactly the same signature as
  a service stub. It will be called before or after an api hook is
  executed, depending on whether this list is for precall of postcall hooks.
  Hooks can be used for debugging purposes (check certain
  pre- or postconditions on api calls) or to apply patches to protocol
  buffers before/after a call gets submitted.
  """

    def __init__(self):
        """Constructor."""
        self.__content = []
        self.__unique_keys = set()

    def __len__(self):
        """Returns the amount of elements in the collection."""
        return self.__content.__len__()

    def __Insert(self, index, key, function, service=None):
        """Appends a hook at a certain position in the list.

    Args:
      index: the index of where to insert the function
      key: a unique key (within the module) for this particular function.
        If something from the same module with the same key is already
        registered, nothing will be added.
      function: the hook to be added.
      service: optional argument that restricts the hook to a particular api

    Returns:
      True if the collection was modified.
    """
        unique_key = (key, inspect.getmodule(function))
        if unique_key in self.__unique_keys:
            return False
        num_args = len(inspect.getargspec(function)[0])
        if inspect.ismethod(function):
            num_args -= 1
        self.__content.insert(index, (key, function, service, num_args))
        self.__unique_keys.add(unique_key)
        return True

    def Append(self, key, function, service=None):
        """Appends a hook at the end of the list.

    Args:
      key: a unique key (within the module) for this particular function.
        If something from the same module with the same key is already
        registered, nothing will be added.
      function: the hook to be added.
      service: optional argument that restricts the hook to a particular api

    Returns:
      True if the collection was modified.
    """
        return self.__Insert(len(self), key, function, service)

    def Push(self, key, function, service=None):
        """Inserts a hook at the beginning of the list.

    Args:
      key: a unique key (within the module) for this particular function.
        If something from the same module with the same key is already
        registered, nothing will be added.
      function: the hook to be added.
      service: optional argument that restricts the hook to a particular api

    Returns:
      True if the collection was modified.
    """
        return self.__Insert(0, key, function, service)

    def Clear(self):
        """Removes all hooks from the list (useful for unit tests)."""
        self.__content = []
        self.__unique_keys = set()

    def Call(self, service, call, request, response, rpc=None, error=None):
        """Invokes all hooks in this collection.

    NOTE: For backwards compatibility, if error is not None, hooks
    with 4 or 5 arguments are *not* called.  This situation
    (error=None) only occurs when the RPC request raised an exception;
    in the past no hooks would be called at all in that case.

    Args:
      service: string representing which service to call
      call: string representing which function to call
      request: protocol buffer for the request
      response: protocol buffer for the response
      rpc: optional RPC used to make this call
      error: optional Exception instance to be passed as 6th argument
    """
        for key, function, srv, num_args in self.__content:
            if srv is None or srv == service:
                if num_args == 6:
                    function(service, call, request, response, rpc, error)
                elif error is not None:
                    pass
                elif num_args == 5:
                    function(service, call, request, response, rpc)
                else:
                    function(service, call, request, response)