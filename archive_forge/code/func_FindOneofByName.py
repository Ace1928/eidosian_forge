import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def FindOneofByName(self, full_name):
    """Loads the named oneof descriptor from the pool.

    Args:
      full_name (str): The full name of the oneof descriptor to load.

    Returns:
      OneofDescriptor: The oneof descriptor for the named oneof.

    Raises:
      KeyError: if the oneof cannot be found in the pool.
    """
    full_name = _NormalizeFullyQualifiedName(full_name)
    message_name, _, oneof_name = full_name.rpartition('.')
    message_descriptor = self.FindMessageTypeByName(message_name)
    return message_descriptor.oneofs_by_name[oneof_name]