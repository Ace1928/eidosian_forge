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
def _AddEqualsMethod(message_descriptor, cls):
    """Helper for _AddMessageMethods()."""

    def __eq__(self, other):
        if not isinstance(other, message_mod.Message) or other.DESCRIPTOR != self.DESCRIPTOR:
            return NotImplemented
        if self is other:
            return True
        if self.DESCRIPTOR.full_name == _AnyFullTypeName:
            any_a = _InternalUnpackAny(self)
            any_b = _InternalUnpackAny(other)
            if any_a and any_b:
                return any_a == any_b
        if not self.ListFields() == other.ListFields():
            return False
        unknown_fields = list(self._unknown_fields)
        unknown_fields.sort()
        other_unknown_fields = list(other._unknown_fields)
        other_unknown_fields.sort()
        return unknown_fields == other_unknown_fields
    cls.__eq__ = __eq__