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
def _AddSerializePartialToStringMethod(message_descriptor, cls):
    """Helper for _AddMessageMethods()."""

    def SerializePartialToString(self, **kwargs):
        out = BytesIO()
        self._InternalSerialize(out.write, **kwargs)
        return out.getvalue()
    cls.SerializePartialToString = SerializePartialToString

    def InternalSerialize(self, write_bytes, deterministic=None):
        if deterministic is None:
            deterministic = api_implementation.IsPythonDefaultSerializationDeterministic()
        else:
            deterministic = bool(deterministic)
        descriptor = self.DESCRIPTOR
        if descriptor._is_map_entry:
            key_field = descriptor.fields_by_name['key']
            _MaybeAddEncoder(cls, key_field)
            key_field._encoder(write_bytes, self.key, deterministic)
            value_field = descriptor.fields_by_name['value']
            _MaybeAddEncoder(cls, value_field)
            value_field._encoder(write_bytes, self.value, deterministic)
        else:
            for field_descriptor, field_value in self.ListFields():
                _MaybeAddEncoder(cls, field_descriptor)
                field_descriptor._encoder(write_bytes, field_value, deterministic)
            for tag_bytes, value_bytes in self._unknown_fields:
                write_bytes(tag_bytes)
                write_bytes(value_bytes)
    cls._InternalSerialize = InternalSerialize