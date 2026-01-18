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
def _AddEnumValues(descriptor, cls):
    """Sets class-level attributes for all enum fields defined in this message.

  Also exporting a class-level object that can name enum values.

  Args:
    descriptor: Descriptor object for this message type.
    cls: Class we're constructing for this message type.
  """
    for enum_type in descriptor.enum_types:
        setattr(cls, enum_type.name, enum_type_wrapper.EnumTypeWrapper(enum_type))
        for enum_value in enum_type.values:
            setattr(cls, enum_value.name, enum_value.number)