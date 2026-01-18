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
def _DiscardUnknownFields(self):
    self._unknown_fields = []
    self._unknown_field_set = None
    for field, value in self.ListFields():
        if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
            if _IsMapField(field):
                if _IsMessageMapField(field):
                    for key in value:
                        value[key].DiscardUnknownFields()
            elif field.label == _FieldDescriptor.LABEL_REPEATED:
                for sub_message in value:
                    sub_message.DiscardUnknownFields()
            else:
                value.DiscardUnknownFields()