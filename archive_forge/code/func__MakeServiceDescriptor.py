import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _MakeServiceDescriptor(self, service_proto, service_index, scope, package, file_desc):
    """Make a protobuf ServiceDescriptor given a ServiceDescriptorProto.

    Args:
      service_proto: The descriptor_pb2.ServiceDescriptorProto protobuf message.
      service_index: The index of the service in the File.
      scope: Dict mapping short and full symbols to message and enum types.
      package: Optional package name for the new message EnumDescriptor.
      file_desc: The file containing the service descriptor.

    Returns:
      The added descriptor.
    """
    if package:
        service_name = '.'.join((package, service_proto.name))
    else:
        service_name = service_proto.name
    methods = [self._MakeMethodDescriptor(method_proto, service_name, package, scope, index) for index, method_proto in enumerate(service_proto.method)]
    desc = descriptor.ServiceDescriptor(name=service_proto.name, full_name=service_name, index=service_index, methods=methods, options=_OptionsOrNone(service_proto), file=file_desc, create_key=descriptor._internal_create_key)
    self._CheckConflictRegister(desc, desc.full_name, desc.file.name)
    self._service_descriptors[service_name] = desc
    return desc