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
def _AddMergeFromMethod(cls):
    LABEL_REPEATED = _FieldDescriptor.LABEL_REPEATED
    CPPTYPE_MESSAGE = _FieldDescriptor.CPPTYPE_MESSAGE

    def MergeFrom(self, msg):
        if not isinstance(msg, cls):
            raise TypeError('Parameter to MergeFrom() must be instance of same class: expected %s got %s.' % (_FullyQualifiedClassName(cls), _FullyQualifiedClassName(msg.__class__)))
        assert msg is not self
        self._Modified()
        fields = self._fields
        for field, value in msg._fields.items():
            if field.label == LABEL_REPEATED:
                field_value = fields.get(field)
                if field_value is None:
                    field_value = field._default_constructor(self)
                    fields[field] = field_value
                field_value.MergeFrom(value)
            elif field.cpp_type == CPPTYPE_MESSAGE:
                if value._is_present_in_parent:
                    field_value = fields.get(field)
                    if field_value is None:
                        field_value = field._default_constructor(self)
                        fields[field] = field_value
                    field_value.MergeFrom(value)
            else:
                self._fields[field] = value
                if field.containing_oneof:
                    self._UpdateOneofState(field)
        if msg._unknown_fields:
            if not self._unknown_fields:
                self._unknown_fields = []
            self._unknown_fields.extend(msg._unknown_fields)
            if self._unknown_field_set is None:
                self._unknown_field_set = containers.UnknownFieldSet()
            self._unknown_field_set._extend(msg._unknown_field_set)
    cls.MergeFrom = MergeFrom