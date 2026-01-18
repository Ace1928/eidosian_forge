import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def FindExtensionByName(self, full_name):
    """Loads the named extension descriptor from the pool.

    Args:
      full_name (str): The full name of the extension descriptor to load.

    Returns:
      FieldDescriptor: The field descriptor for the named extension.

    Raises:
      KeyError: if the extension cannot be found in the pool.
    """
    full_name = _NormalizeFullyQualifiedName(full_name)
    try:
        return self._toplevel_extensions[full_name]
    except KeyError:
        pass
    message_name, _, extension_name = full_name.rpartition('.')
    try:
        scope = self.FindMessageTypeByName(message_name)
    except KeyError:
        scope = self._FindFileContainingSymbolInDb(full_name)
    return scope.extensions_by_name[extension_name]