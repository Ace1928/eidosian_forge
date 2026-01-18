import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def FindMessageTypeByName(self, full_name):
    """Loads the named descriptor from the pool.

    Args:
      full_name (str): The full name of the descriptor to load.

    Returns:
      Descriptor: The descriptor for the named type.

    Raises:
      KeyError: if the message cannot be found in the pool.
    """
    full_name = _NormalizeFullyQualifiedName(full_name)
    if full_name not in self._descriptors:
        self._FindFileContainingSymbolInDb(full_name)
    return self._descriptors[full_name]