import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _AddDescriptor(self, desc):
    """Adds a Descriptor to the pool, non-recursively.

    If the Descriptor contains nested messages or enums, the caller must
    explicitly register them. This method also registers the FileDescriptor
    associated with the message.

    Args:
      desc: A Descriptor.
    """
    if not isinstance(desc, descriptor.Descriptor):
        raise TypeError('Expected instance of descriptor.Descriptor.')
    self._CheckConflictRegister(desc, desc.full_name, desc.file.name)
    self._descriptors[desc.full_name] = desc
    self._AddFileDescriptor(desc.file)