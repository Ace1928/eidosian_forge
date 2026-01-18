import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def FindEnumTypeByName(self, full_name):
    """Loads the named enum descriptor from the pool.

    Args:
      full_name (str): The full name of the enum descriptor to load.

    Returns:
      EnumDescriptor: The enum descriptor for the named type.

    Raises:
      KeyError: if the enum cannot be found in the pool.
    """
    full_name = _NormalizeFullyQualifiedName(full_name)
    if full_name not in self._enum_descriptors:
        self._FindFileContainingSymbolInDb(full_name)
    return self._enum_descriptors[full_name]