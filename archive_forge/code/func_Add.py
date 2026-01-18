import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def Add(self, file_desc_proto):
    """Adds the FileDescriptorProto and its types to this pool.

    Args:
      file_desc_proto (FileDescriptorProto): The file descriptor to add.
    """
    self._internal_db.Add(file_desc_proto)