from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
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