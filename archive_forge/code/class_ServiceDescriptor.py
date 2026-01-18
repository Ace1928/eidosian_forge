import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
class ServiceDescriptor(_NestedDescriptorBase):
    """Descriptor for a service.

  Attributes:
    name (str): Name of the service.
    full_name (str): Full name of the service, including package name.
    index (int): 0-indexed index giving the order that this services
      definition appears within the .proto file.
    methods (list[MethodDescriptor]): List of methods provided by this
      service.
    methods_by_name (dict(str, MethodDescriptor)): Same
      :class:`MethodDescriptor` objects as in :attr:`methods_by_name`, but
      indexed by "name" attribute in each :class:`MethodDescriptor`.
    options (descriptor_pb2.ServiceOptions): Service options message or
      None to use default service options.
    file (FileDescriptor): Reference to file info.
  """
    if _USE_C_DESCRIPTORS:
        _C_DESCRIPTOR_CLASS = _message.ServiceDescriptor

        def __new__(cls, name=None, full_name=None, index=None, methods=None, options=None, serialized_options=None, file=None, serialized_start=None, serialized_end=None, create_key=None):
            _message.Message._CheckCalledFromGeneratedFile()
            return _message.default_pool.FindServiceByName(full_name)

    def __init__(self, name, full_name, index, methods, options=None, serialized_options=None, file=None, serialized_start=None, serialized_end=None, create_key=None):
        if create_key is not _internal_create_key:
            _Deprecated('ServiceDescriptor')
        super(ServiceDescriptor, self).__init__(options, 'ServiceOptions', name, full_name, file, None, serialized_start=serialized_start, serialized_end=serialized_end, serialized_options=serialized_options)
        self.index = index
        self.methods = methods
        self.methods_by_name = dict(((m.name, m) for m in methods))
        for method in self.methods:
            method.file = self.file
            method.containing_service = self

    @property
    def _parent(self):
        return self.file

    def FindMethodByName(self, name):
        """Searches for the specified method, and returns its descriptor.

    Args:
      name (str): Name of the method.

    Returns:
      MethodDescriptor: The descriptor for the requested method.

    Raises:
      KeyError: if the method cannot be found in the service.
    """
        return self.methods_by_name[name]

    def CopyToProto(self, proto):
        """Copies this to a descriptor_pb2.ServiceDescriptorProto.

    Args:
      proto (descriptor_pb2.ServiceDescriptorProto): An empty descriptor proto.
    """
        super(ServiceDescriptor, self).CopyToProto(proto)