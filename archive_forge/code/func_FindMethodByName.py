import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
def FindMethodByName(self, name):
    """Searches for the specified method, and returns its descriptor.

    Args:
      name (str): Name of the method.

    Returns:
      MethodDescriptor: The descriptor for the requested method.

    Raises:
      KeyError: if the method cannot be found in the service.
    """
    return self.methods_by_name[name]