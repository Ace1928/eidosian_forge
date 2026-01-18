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
def _ConvertMapFieldValue(self, value, message, field, path):
    """Convert map field value for a message map field.

    Args:
      value: A JSON object to convert the map field value.
      message: A protocol message to record the converted data.
      field: The descriptor of the map field to be converted.
      path: parent path to log parse error info.

    Raises:
      ParseError: In case of convert problems.
    """
    if not isinstance(value, dict):
        raise ParseError('Map field {0} must be in a dict which is {1} at {2}'.format(field.name, value, path))
    key_field = field.message_type.fields_by_name['key']
    value_field = field.message_type.fields_by_name['value']
    for key in value:
        key_value = _ConvertScalarFieldValue(key, key_field, '{0}.key'.format(path), True)
        if value_field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
            self.ConvertMessage(value[key], getattr(message, field.name)[key_value], '{0}[{1}]'.format(path, key_value))
        else:
            getattr(message, field.name)[key_value] = _ConvertScalarFieldValue(value[key], value_field, path='{0}[{1}]'.format(path, key_value))