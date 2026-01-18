import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def FindFieldByName(self, full_name):
    """Loads the named field descriptor from the pool.

    Args:
      full_name (str): The full name of the field descriptor to load.

    Returns:
      FieldDescriptor: The field descriptor for the named field.

    Raises:
      KeyError: if the field cannot be found in the pool.
    """
    full_name = _NormalizeFullyQualifiedName(full_name)
    message_name, _, field_name = full_name.rpartition('.')
    message_descriptor = self.FindMessageTypeByName(message_name)
    return message_descriptor.fields_by_name[field_name]