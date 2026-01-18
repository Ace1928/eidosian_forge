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
def IsInitialized(self, errors=None):
    """Checks if all required fields of a message are set.

    Args:
      errors:  A list which, if provided, will be populated with the field
               paths of all missing required fields.

    Returns:
      True iff the specified message has all required fields set.
    """
    for field in required_fields:
        if field not in self._fields or (field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE and (not self._fields[field]._is_present_in_parent)):
            if errors is not None:
                errors.extend(self.FindInitializationErrors())
            return False
    for field, value in list(self._fields.items()):
        if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
            if field.label == _FieldDescriptor.LABEL_REPEATED:
                if field.message_type._is_map_entry:
                    continue
                for element in value:
                    if not element.IsInitialized():
                        if errors is not None:
                            errors.extend(self.FindInitializationErrors())
                        return False
            elif value._is_present_in_parent and (not value.IsInitialized()):
                if errors is not None:
                    errors.extend(self.FindInitializationErrors())
                return False
    return True