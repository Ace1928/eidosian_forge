import base64
from collections import OrderedDict
import json
import math
from operator import methodcaller
import re
from google.protobuf import descriptor
from google.protobuf import message_factory
from google.protobuf import symbol_database
from google.protobuf.internal import type_checkers
def _FieldToJsonObject(self, field, value):
    """Converts field value according to Proto3 JSON Specification."""
    if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
        return self._MessageToJsonObject(value)
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_ENUM:
        if self.use_integers_for_enums:
            return value
        if field.enum_type.full_name == 'google.protobuf.NullValue':
            return None
        enum_value = field.enum_type.values_by_number.get(value, None)
        if enum_value is not None:
            return enum_value.name
        elif field.enum_type.is_closed:
            raise SerializeToJsonError('Enum field contains an integer value which can not mapped to an enum value.')
        else:
            return value
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_STRING:
        if field.type == descriptor.FieldDescriptor.TYPE_BYTES:
            return base64.b64encode(value).decode('utf-8')
        else:
            return value
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_BOOL:
        return bool(value)
    elif field.cpp_type in _INT64_TYPES:
        return str(value)
    elif field.cpp_type in _FLOAT_TYPES:
        if math.isinf(value):
            if value < 0.0:
                return _NEG_INFINITY
            else:
                return _INFINITY
        if math.isnan(value):
            return _NAN
        if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_FLOAT:
            if self.float_format:
                return float(format(value, self.float_format))
            else:
                return type_checkers.ToShortestFloat(value)
    return value