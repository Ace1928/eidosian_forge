import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
class FileDescriptor(DescriptorBase):
    """Descriptor for a file. Mimics the descriptor_pb2.FileDescriptorProto.

  Note that :attr:`enum_types_by_name`, :attr:`extensions_by_name`, and
  :attr:`dependencies` fields are only set by the
  :py:mod:`google.protobuf.message_factory` module, and not by the generated
  proto code.

  Attributes:
    name (str): Name of file, relative to root of source tree.
    package (str): Name of the package
    edition (Edition): Enum value indicating edition of the file
    serialized_pb (bytes): Byte string of serialized
      :class:`descriptor_pb2.FileDescriptorProto`.
    dependencies (list[FileDescriptor]): List of other :class:`FileDescriptor`
      objects this :class:`FileDescriptor` depends on.
    public_dependencies (list[FileDescriptor]): A subset of
      :attr:`dependencies`, which were declared as "public".
    message_types_by_name (dict(str, Descriptor)): Mapping from message names to
      their :class:`Descriptor`.
    enum_types_by_name (dict(str, EnumDescriptor)): Mapping from enum names to
      their :class:`EnumDescriptor`.
    extensions_by_name (dict(str, FieldDescriptor)): Mapping from extension
      names declared at file scope to their :class:`FieldDescriptor`.
    services_by_name (dict(str, ServiceDescriptor)): Mapping from services'
      names to their :class:`ServiceDescriptor`.
    pool (DescriptorPool): The pool this descriptor belongs to.  When not passed
      to the constructor, the global default pool is used.
  """
    if _USE_C_DESCRIPTORS:
        _C_DESCRIPTOR_CLASS = _message.FileDescriptor

        def __new__(cls, name, package, options=None, serialized_options=None, serialized_pb=None, dependencies=None, public_dependencies=None, syntax=None, edition=None, pool=None, create_key=None):
            if serialized_pb:
                return _message.default_pool.AddSerializedFile(serialized_pb)
            else:
                return super(FileDescriptor, cls).__new__(cls)

    def __init__(self, name, package, options=None, serialized_options=None, serialized_pb=None, dependencies=None, public_dependencies=None, syntax=None, edition=None, pool=None, create_key=None):
        """Constructor."""
        if create_key is not _internal_create_key:
            _Deprecated('FileDescriptor')
        super(FileDescriptor, self).__init__(self, options, serialized_options, 'FileOptions')
        if edition and edition != 'EDITION_UNKNOWN':
            self._edition = edition
        elif syntax == 'proto3':
            self._edition = 'EDITION_PROTO3'
        else:
            self._edition = 'EDITION_PROTO2'
        if pool is None:
            from google.protobuf import descriptor_pool
            pool = descriptor_pool.Default()
        self.pool = pool
        self.message_types_by_name = {}
        self.name = name
        self.package = package
        self.serialized_pb = serialized_pb
        self.enum_types_by_name = {}
        self.extensions_by_name = {}
        self.services_by_name = {}
        self.dependencies = dependencies or []
        self.public_dependencies = public_dependencies or []

    def CopyToProto(self, proto):
        """Copies this to a descriptor_pb2.FileDescriptorProto.

    Args:
      proto: An empty descriptor_pb2.FileDescriptorProto.
    """
        proto.ParseFromString(self.serialized_pb)

    @property
    def _parent(self):
        return None