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
def FindInitializationErrors(self):
    """Finds required fields which are not initialized.

    Returns:
      A list of strings.  Each string is a path to an uninitialized field from
      the top-level message, e.g. "foo.bar[5].baz".
    """
    errors = []
    for field in required_fields:
        if not self.HasField(field.name):
            errors.append(field.name)
    for field, value in self.ListFields():
        if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
            if field.is_extension:
                name = '(%s)' % field.full_name
            else:
                name = field.name
            if _IsMapField(field):
                if _IsMessageMapField(field):
                    for key in value:
                        element = value[key]
                        prefix = '%s[%s].' % (name, key)
                        sub_errors = element.FindInitializationErrors()
                        errors += [prefix + error for error in sub_errors]
                else:
                    pass
            elif field.label == _FieldDescriptor.LABEL_REPEATED:
                for i in range(len(value)):
                    element = value[i]
                    prefix = '%s[%d].' % (name, i)
                    sub_errors = element.FindInitializationErrors()
                    errors += [prefix + error for error in sub_errors]
            else:
                prefix = name + '.'
                sub_errors = value.FindInitializationErrors()
                errors += [prefix + error for error in sub_errors]
    return errors