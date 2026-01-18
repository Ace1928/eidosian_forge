import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
class EnumDescriptor(_NestedDescriptorBase):
    """Descriptor for an enum defined in a .proto file.

  Attributes:
    name (str): Name of the enum type.
    full_name (str): Full name of the type, including package name
      and any enclosing type(s).

    values (list[EnumValueDescriptor]): List of the values
      in this enum.
    values_by_name (dict(str, EnumValueDescriptor)): Same as :attr:`values`,
      but indexed by the "name" field of each EnumValueDescriptor.
    values_by_number (dict(int, EnumValueDescriptor)): Same as :attr:`values`,
      but indexed by the "number" field of each EnumValueDescriptor.
    containing_type (Descriptor): Descriptor of the immediate containing
      type of this enum, or None if this is an enum defined at the
      top level in a .proto file.  Set by Descriptor's constructor
      if we're passed into one.
    file (FileDescriptor): Reference to file descriptor.
    options (descriptor_pb2.EnumOptions): Enum options message or
      None to use default enum options.
  """
    if _USE_C_DESCRIPTORS:
        _C_DESCRIPTOR_CLASS = _message.EnumDescriptor

        def __new__(cls, name, full_name, filename, values, containing_type=None, options=None, serialized_options=None, file=None, serialized_start=None, serialized_end=None, create_key=None):
            _message.Message._CheckCalledFromGeneratedFile()
            return _message.default_pool.FindEnumTypeByName(full_name)

    def __init__(self, name, full_name, filename, values, containing_type=None, options=None, serialized_options=None, file=None, serialized_start=None, serialized_end=None, create_key=None):
        """Arguments are as described in the attribute description above.

    Note that filename is an obsolete argument, that is not used anymore.
    Please use file.name to access this as an attribute.
    """
        if create_key is not _internal_create_key:
            _Deprecated('EnumDescriptor')
        super(EnumDescriptor, self).__init__(options, 'EnumOptions', name, full_name, file, containing_type, serialized_start=serialized_start, serialized_end=serialized_end, serialized_options=serialized_options)
        self.values = values
        for value in self.values:
            value.file = file
            value.type = self
        self.values_by_name = dict(((v.name, v) for v in values))
        self.values_by_number = dict(((v.number, v) for v in reversed(values)))

    @property
    def _parent(self):
        return self.containing_type or self.file

    @property
    def is_closed(self):
        """Returns true whether this is a "closed" enum.

    This means that it:
    - Has a fixed set of values, rather than being equivalent to an int32.
    - Encountering values not in this set causes them to be treated as unknown
      fields.
    - The first value (i.e., the default) may be nonzero.

    WARNING: Some runtimes currently have a quirk where non-closed enums are
    treated as closed when used as the type of fields defined in a
    `syntax = proto2;` file. This quirk is not present in all runtimes; as of
    writing, we know that:

    - C++, Java, and C++-based Python share this quirk.
    - UPB and UPB-based Python do not.
    - PHP and Ruby treat all enums as open regardless of declaration.

    Care should be taken when using this function to respect the target
    runtime's enum handling quirks.
    """
        return self._GetFeatures().enum_type == _FEATURESET_ENUM_TYPE_CLOSED

    def CopyToProto(self, proto):
        """Copies this to a descriptor_pb2.EnumDescriptorProto.

    Args:
      proto (descriptor_pb2.EnumDescriptorProto): An empty descriptor proto.
    """
        super(EnumDescriptor, self).CopyToProto(proto)