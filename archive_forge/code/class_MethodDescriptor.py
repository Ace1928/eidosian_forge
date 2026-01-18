import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
class MethodDescriptor(DescriptorBase):
    """Descriptor for a method in a service.

  Attributes:
    name (str): Name of the method within the service.
    full_name (str): Full name of method.
    index (int): 0-indexed index of the method inside the service.
    containing_service (ServiceDescriptor): The service that contains this
      method.
    input_type (Descriptor): The descriptor of the message that this method
      accepts.
    output_type (Descriptor): The descriptor of the message that this method
      returns.
    client_streaming (bool): Whether this method uses client streaming.
    server_streaming (bool): Whether this method uses server streaming.
    options (descriptor_pb2.MethodOptions or None): Method options message, or
      None to use default method options.
  """
    if _USE_C_DESCRIPTORS:
        _C_DESCRIPTOR_CLASS = _message.MethodDescriptor

        def __new__(cls, name, full_name, index, containing_service, input_type, output_type, client_streaming=False, server_streaming=False, options=None, serialized_options=None, create_key=None):
            _message.Message._CheckCalledFromGeneratedFile()
            return _message.default_pool.FindMethodByName(full_name)

    def __init__(self, name, full_name, index, containing_service, input_type, output_type, client_streaming=False, server_streaming=False, options=None, serialized_options=None, create_key=None):
        """The arguments are as described in the description of MethodDescriptor
    attributes above.

    Note that containing_service may be None, and may be set later if necessary.
    """
        if create_key is not _internal_create_key:
            _Deprecated('MethodDescriptor')
        super(MethodDescriptor, self).__init__(containing_service.file if containing_service else None, options, serialized_options, 'MethodOptions')
        self.name = name
        self.full_name = full_name
        self.index = index
        self.containing_service = containing_service
        self.input_type = input_type
        self.output_type = output_type
        self.client_streaming = client_streaming
        self.server_streaming = server_streaming

    @property
    def _parent(self):
        return self.containing_service

    def CopyToProto(self, proto):
        """Copies this to a descriptor_pb2.MethodDescriptorProto.

    Args:
      proto (descriptor_pb2.MethodDescriptorProto): An empty descriptor proto.

    Raises:
      Error: If self couldn't be serialized, due to too few constructor
        arguments.
    """
        if self.containing_service is not None:
            from google.protobuf import descriptor_pb2
            service_proto = descriptor_pb2.ServiceDescriptorProto()
            self.containing_service.CopyToProto(service_proto)
            proto.CopyFrom(service_proto.method[self.index])
        else:
            raise Error('Descriptor does not contain a service.')