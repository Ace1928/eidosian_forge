import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
class Descriptor(_NestedDescriptorBase):
    """Descriptor for a protocol message type.

  Attributes:
      name (str): Name of this protocol message type.
      full_name (str): Fully-qualified name of this protocol message type,
          which will include protocol "package" name and the name of any
          enclosing types.
      containing_type (Descriptor): Reference to the descriptor of the type
          containing us, or None if this is top-level.
      fields (list[FieldDescriptor]): Field descriptors for all fields in
          this type.
      fields_by_number (dict(int, FieldDescriptor)): Same
          :class:`FieldDescriptor` objects as in :attr:`fields`, but indexed
          by "number" attribute in each FieldDescriptor.
      fields_by_name (dict(str, FieldDescriptor)): Same
          :class:`FieldDescriptor` objects as in :attr:`fields`, but indexed by
          "name" attribute in each :class:`FieldDescriptor`.
      nested_types (list[Descriptor]): Descriptor references
          for all protocol message types nested within this one.
      nested_types_by_name (dict(str, Descriptor)): Same Descriptor
          objects as in :attr:`nested_types`, but indexed by "name" attribute
          in each Descriptor.
      enum_types (list[EnumDescriptor]): :class:`EnumDescriptor` references
          for all enums contained within this type.
      enum_types_by_name (dict(str, EnumDescriptor)): Same
          :class:`EnumDescriptor` objects as in :attr:`enum_types`, but
          indexed by "name" attribute in each EnumDescriptor.
      enum_values_by_name (dict(str, EnumValueDescriptor)): Dict mapping
          from enum value name to :class:`EnumValueDescriptor` for that value.
      extensions (list[FieldDescriptor]): All extensions defined directly
          within this message type (NOT within a nested type).
      extensions_by_name (dict(str, FieldDescriptor)): Same FieldDescriptor
          objects as :attr:`extensions`, but indexed by "name" attribute of each
          FieldDescriptor.
      is_extendable (bool):  Does this type define any extension ranges?
      oneofs (list[OneofDescriptor]): The list of descriptors for oneof fields
          in this message.
      oneofs_by_name (dict(str, OneofDescriptor)): Same objects as in
          :attr:`oneofs`, but indexed by "name" attribute.
      file (FileDescriptor): Reference to file descriptor.
      is_map_entry: If the message type is a map entry.

  """
    if _USE_C_DESCRIPTORS:
        _C_DESCRIPTOR_CLASS = _message.Descriptor

        def __new__(cls, name=None, full_name=None, filename=None, containing_type=None, fields=None, nested_types=None, enum_types=None, extensions=None, options=None, serialized_options=None, is_extendable=True, extension_ranges=None, oneofs=None, file=None, serialized_start=None, serialized_end=None, syntax=None, is_map_entry=False, create_key=None):
            _message.Message._CheckCalledFromGeneratedFile()
            return _message.default_pool.FindMessageTypeByName(full_name)

    def __init__(self, name, full_name, filename, containing_type, fields, nested_types, enum_types, extensions, options=None, serialized_options=None, is_extendable=True, extension_ranges=None, oneofs=None, file=None, serialized_start=None, serialized_end=None, syntax=None, is_map_entry=False, create_key=None):
        """Arguments to __init__() are as described in the description
    of Descriptor fields above.

    Note that filename is an obsolete argument, that is not used anymore.
    Please use file.name to access this as an attribute.
    """
        if create_key is not _internal_create_key:
            _Deprecated('Descriptor')
        super(Descriptor, self).__init__(options, 'MessageOptions', name, full_name, file, containing_type, serialized_start=serialized_start, serialized_end=serialized_end, serialized_options=serialized_options)
        self.fields = fields
        for field in self.fields:
            field.containing_type = self
            field.file = file
        self.fields_by_number = dict(((f.number, f) for f in fields))
        self.fields_by_name = dict(((f.name, f) for f in fields))
        self._fields_by_camelcase_name = None
        self.nested_types = nested_types
        for nested_type in nested_types:
            nested_type.containing_type = self
        self.nested_types_by_name = dict(((t.name, t) for t in nested_types))
        self.enum_types = enum_types
        for enum_type in self.enum_types:
            enum_type.containing_type = self
        self.enum_types_by_name = dict(((t.name, t) for t in enum_types))
        self.enum_values_by_name = dict(((v.name, v) for t in enum_types for v in t.values))
        self.extensions = extensions
        for extension in self.extensions:
            extension.extension_scope = self
        self.extensions_by_name = dict(((f.name, f) for f in extensions))
        self.is_extendable = is_extendable
        self.extension_ranges = extension_ranges
        self.oneofs = oneofs if oneofs is not None else []
        self.oneofs_by_name = dict(((o.name, o) for o in self.oneofs))
        for oneof in self.oneofs:
            oneof.containing_type = self
            oneof.file = file
        self._is_map_entry = is_map_entry

    @property
    def _parent(self):
        return self.containing_type or self.file

    @property
    def fields_by_camelcase_name(self):
        """Same FieldDescriptor objects as in :attr:`fields`, but indexed by
    :attr:`FieldDescriptor.camelcase_name`.
    """
        if self._fields_by_camelcase_name is None:
            self._fields_by_camelcase_name = dict(((f.camelcase_name, f) for f in self.fields))
        return self._fields_by_camelcase_name

    def EnumValueName(self, enum, value):
        """Returns the string name of an enum value.

    This is just a small helper method to simplify a common operation.

    Args:
      enum: string name of the Enum.
      value: int, value of the enum.

    Returns:
      string name of the enum value.

    Raises:
      KeyError if either the Enum doesn't exist or the value is not a valid
        value for the enum.
    """
        return self.enum_types_by_name[enum].values_by_number[value].name

    def CopyToProto(self, proto):
        """Copies this to a descriptor_pb2.DescriptorProto.

    Args:
      proto: An empty descriptor_pb2.DescriptorProto.
    """
        super(Descriptor, self).CopyToProto(proto)