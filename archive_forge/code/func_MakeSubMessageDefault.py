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
def MakeSubMessageDefault(message):
    if not hasattr(message_type, '_concrete_class'):
        from google.protobuf import message_factory
        message_factory.GetMessageClass(message_type)
    result = message_type._concrete_class()
    result._SetListener(_OneofListener(message, field) if field.containing_oneof is not None else message._listener_for_children)
    return result