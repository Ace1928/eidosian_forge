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
def _AddClearFieldMethod(message_descriptor, cls):
    """Helper for _AddMessageMethods()."""

    def ClearField(self, field_name):
        try:
            field = message_descriptor.fields_by_name[field_name]
        except KeyError:
            try:
                field = message_descriptor.oneofs_by_name[field_name]
                if field in self._oneofs:
                    field = self._oneofs[field]
                else:
                    return
            except KeyError:
                raise ValueError('Protocol message %s has no "%s" field.' % (message_descriptor.name, field_name))
        if field in self._fields:
            if hasattr(self._fields[field], 'InvalidateIterators'):
                self._fields[field].InvalidateIterators()
            del self._fields[field]
            if self._oneofs.get(field.containing_oneof, None) is field:
                del self._oneofs[field.containing_oneof]
        self._Modified()
    cls.ClearField = ClearField