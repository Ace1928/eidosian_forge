import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
def GetOptions(self):
    """Retrieves descriptor options.

    Returns:
      The options set on this descriptor.
    """
    if not self._loaded_options:
        self._LazyLoadOptions()
    return self._loaded_options