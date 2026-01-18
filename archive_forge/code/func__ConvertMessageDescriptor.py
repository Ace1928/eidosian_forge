import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _ConvertMessageDescriptor(self, desc_proto, package=None, file_desc=None, scope=None, syntax=None):
    """Adds the proto to the pool in the specified package.

    Args:
      desc_proto: The descriptor_pb2.DescriptorProto protobuf message.
      package: The package the proto should be located in.
      file_desc: The file containing this message.
      scope: Dict mapping short and full symbols to message and enum types.
      syntax: string indicating syntax of the file ("proto2" or "proto3")

    Returns:
      The added descriptor.
    """
    if package:
        desc_name = '.'.join((package, desc_proto.name))
    else:
        desc_name = desc_proto.name
    if file_desc is None:
        file_name = None
    else:
        file_name = file_desc.name
    if scope is None:
        scope = {}
    nested = [self._ConvertMessageDescriptor(nested, desc_name, file_desc, scope, syntax) for nested in desc_proto.nested_type]
    enums = [self._ConvertEnumDescriptor(enum, desc_name, file_desc, None, scope, False) for enum in desc_proto.enum_type]
    fields = [self._MakeFieldDescriptor(field, desc_name, index, file_desc) for index, field in enumerate(desc_proto.field)]
    extensions = [self._MakeFieldDescriptor(extension, desc_name, index, file_desc, is_extension=True) for index, extension in enumerate(desc_proto.extension)]
    oneofs = [descriptor.OneofDescriptor(desc.name, '.'.join((desc_name, desc.name)), index, None, [], _OptionsOrNone(desc), create_key=descriptor._internal_create_key) for index, desc in enumerate(desc_proto.oneof_decl)]
    extension_ranges = [(r.start, r.end) for r in desc_proto.extension_range]
    if extension_ranges:
        is_extendable = True
    else:
        is_extendable = False
    desc = descriptor.Descriptor(name=desc_proto.name, full_name=desc_name, filename=file_name, containing_type=None, fields=fields, oneofs=oneofs, nested_types=nested, enum_types=enums, extensions=extensions, options=_OptionsOrNone(desc_proto), is_extendable=is_extendable, extension_ranges=extension_ranges, file=file_desc, serialized_start=None, serialized_end=None, is_map_entry=desc_proto.options.map_entry, create_key=descriptor._internal_create_key)
    for nested in desc.nested_types:
        nested.containing_type = desc
    for enum in desc.enum_types:
        enum.containing_type = desc
    for field_index, field_desc in enumerate(desc_proto.field):
        if field_desc.HasField('oneof_index'):
            oneof_index = field_desc.oneof_index
            oneofs[oneof_index].fields.append(fields[field_index])
            fields[field_index].containing_oneof = oneofs[oneof_index]
    scope[_PrefixWithDot(desc_name)] = desc
    self._CheckConflictRegister(desc, desc.full_name, desc.file.name)
    self._descriptors[desc_name] = desc
    return desc