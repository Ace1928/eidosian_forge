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
def _AddPropertiesForExtensions(descriptor, cls):
    """Adds properties for all fields in this protocol message type."""
    extensions = descriptor.extensions_by_name
    for extension_name, extension_field in extensions.items():
        constant_name = extension_name.upper() + '_FIELD_NUMBER'
        setattr(cls, constant_name, extension_field.number)
    if descriptor.file is not None:
        pool = descriptor.file.pool