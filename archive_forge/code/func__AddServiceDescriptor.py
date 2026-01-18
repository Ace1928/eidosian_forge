import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _AddServiceDescriptor(self, service_desc):
    """Adds a ServiceDescriptor to the pool.

    Args:
      service_desc: A ServiceDescriptor.
    """
    if not isinstance(service_desc, descriptor.ServiceDescriptor):
        raise TypeError('Expected instance of descriptor.ServiceDescriptor.')
    self._CheckConflictRegister(service_desc, service_desc.full_name, service_desc.file.name)
    self._service_descriptors[service_desc.full_name] = service_desc