from io import BytesIO
import struct
import sys
import warnings
import weakref
from google.protobuf import descriptor as descriptor_mod
from google.protobuf import message as message_mod
from google.protobuf import text_format
from google.protobuf.internal import api_implementation
from google.protobuf.internal import containers
from google.protobuf.internal import decoder
from google.protobuf.internal import encoder
from google.protobuf.internal import enum_type_wrapper
from google.protobuf.internal import extension_dict
from google.protobuf.internal import message_listener as message_listener_mod
from google.protobuf.internal import type_checkers
from google.protobuf.internal import well_known_types
from google.protobuf.internal import wire_format
def _AddHasFieldMethod(message_descriptor, cls):
    """Helper for _AddMessageMethods()."""
    hassable_fields = {}
    for field in message_descriptor.fields:
        if field.label == _FieldDescriptor.LABEL_REPEATED:
            continue
        if not field.has_presence:
            continue
        hassable_fields[field.name] = field
    for oneof in message_descriptor.oneofs:
        hassable_fields[oneof.name] = oneof

    def HasField(self, field_name):
        try:
            field = hassable_fields[field_name]
        except KeyError as exc:
            raise ValueError('Protocol message %s has no non-repeated field "%s" nor has presence is not available for this field.' % (message_descriptor.full_name, field_name)) from exc
        if isinstance(field, descriptor_mod.OneofDescriptor):
            try:
                return HasField(self, self._oneofs[field].name)
            except KeyError:
                return False
        elif field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
            value = self._fields.get(field)
            return value is not None and value._is_present_in_parent
        else:
            return field in self._fields
    cls.HasField = HasField