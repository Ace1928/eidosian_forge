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
def _MaybeAddEncoder(cls, field_descriptor):
    if hasattr(field_descriptor, '_encoder'):
        return
    is_repeated = field_descriptor.label == _FieldDescriptor.LABEL_REPEATED
    is_map_entry = _IsMapField(field_descriptor)
    is_packed = field_descriptor.is_packed
    if is_map_entry:
        field_encoder = encoder.MapEncoder(field_descriptor)
        sizer = encoder.MapSizer(field_descriptor, _IsMessageMapField(field_descriptor))
    elif _IsMessageSetExtension(field_descriptor):
        field_encoder = encoder.MessageSetItemEncoder(field_descriptor.number)
        sizer = encoder.MessageSetItemSizer(field_descriptor.number)
    else:
        field_encoder = type_checkers.TYPE_TO_ENCODER[field_descriptor.type](field_descriptor.number, is_repeated, is_packed)
        sizer = type_checkers.TYPE_TO_SIZER[field_descriptor.type](field_descriptor.number, is_repeated, is_packed)
    field_descriptor._sizer = sizer
    field_descriptor._encoder = field_encoder