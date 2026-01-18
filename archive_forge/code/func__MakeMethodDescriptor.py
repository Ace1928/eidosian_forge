import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _MakeMethodDescriptor(self, method_proto, service_name, package, scope, index):
    """Creates a method descriptor from a MethodDescriptorProto.

    Args:
      method_proto: The proto describing the method.
      service_name: The name of the containing service.
      package: Optional package name to look up for types.
      scope: Scope containing available types.
      index: Index of the method in the service.

    Returns:
      An initialized MethodDescriptor object.
    """
    full_name = '.'.join((service_name, method_proto.name))
    input_type = self._GetTypeFromScope(package, method_proto.input_type, scope)
    output_type = self._GetTypeFromScope(package, method_proto.output_type, scope)
    return descriptor.MethodDescriptor(name=method_proto.name, full_name=full_name, index=index, containing_service=None, input_type=input_type, output_type=output_type, client_streaming=method_proto.client_streaming, server_streaming=method_proto.server_streaming, options=_OptionsOrNone(method_proto), create_key=descriptor._internal_create_key)