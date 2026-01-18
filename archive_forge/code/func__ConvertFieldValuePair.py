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
def _ConvertFieldValuePair(self, js, message, path):
    """Convert field value pairs into regular message.

    Args:
      js: A JSON object to convert the field value pairs.
      message: A regular protocol message to record the data.
      path: parent path to log parse error info.

    Raises:
      ParseError: In case of problems converting.
    """
    names = []
    message_descriptor = message.DESCRIPTOR
    fields_by_json_name = dict(((f.json_name, f) for f in message_descriptor.fields))
    for name in js:
        try:
            field = fields_by_json_name.get(name, None)
            if not field:
                field = message_descriptor.fields_by_name.get(name, None)
            if not field and _VALID_EXTENSION_NAME.match(name):
                if not message_descriptor.is_extendable:
                    raise ParseError('Message type {0} does not have extensions at {1}'.format(message_descriptor.full_name, path))
                identifier = name[1:-1]
                field = message.Extensions._FindExtensionByName(identifier)
                if not field:
                    identifier = '.'.join(identifier.split('.')[:-1])
                    field = message.Extensions._FindExtensionByName(identifier)
            if not field:
                if self.ignore_unknown_fields:
                    continue
                raise ParseError('Message type "{0}" has no field named "{1}" at "{2}".\n Available Fields(except extensions): "{3}"'.format(message_descriptor.full_name, name, path, [f.json_name for f in message_descriptor.fields]))
            if name in names:
                raise ParseError('Message type "{0}" should not have multiple "{1}" fields at "{2}".'.format(message.DESCRIPTOR.full_name, name, path))
            names.append(name)
            value = js[name]
            if field.containing_oneof is not None and value is not None:
                oneof_name = field.containing_oneof.name
                if oneof_name in names:
                    raise ParseError('Message type "{0}" should not have multiple "{1}" oneof fields at "{2}".'.format(message.DESCRIPTOR.full_name, oneof_name, path))
                names.append(oneof_name)
            if value is None:
                if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE and field.message_type.full_name == 'google.protobuf.Value':
                    sub_message = getattr(message, field.name)
                    sub_message.null_value = 0
                elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_ENUM and field.enum_type.full_name == 'google.protobuf.NullValue':
                    setattr(message, field.name, 0)
                else:
                    message.ClearField(field.name)
                continue
            if _IsMapEntry(field):
                message.ClearField(field.name)
                self._ConvertMapFieldValue(value, message, field, '{0}.{1}'.format(path, name))
            elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
                message.ClearField(field.name)
                if not isinstance(value, list):
                    raise ParseError('repeated field {0} must be in [] which is {1} at {2}'.format(name, value, path))
                if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
                    for index, item in enumerate(value):
                        sub_message = getattr(message, field.name).add()
                        if item is None and sub_message.DESCRIPTOR.full_name != 'google.protobuf.Value':
                            raise ParseError('null is not allowed to be used as an element in a repeated field at {0}.{1}[{2}]'.format(path, name, index))
                        self.ConvertMessage(item, sub_message, '{0}.{1}[{2}]'.format(path, name, index))
                else:
                    for index, item in enumerate(value):
                        if item is None:
                            raise ParseError('null is not allowed to be used as an element in a repeated field at {0}.{1}[{2}]'.format(path, name, index))
                        getattr(message, field.name).append(_ConvertScalarFieldValue(item, field, '{0}.{1}[{2}]'.format(path, name, index)))
            elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
                if field.is_extension:
                    sub_message = message.Extensions[field]
                else:
                    sub_message = getattr(message, field.name)
                sub_message.SetInParent()
                self.ConvertMessage(value, sub_message, '{0}.{1}'.format(path, name))
            elif field.is_extension:
                message.Extensions[field] = _ConvertScalarFieldValue(value, field, '{0}.{1}'.format(path, name))
            else:
                setattr(message, field.name, _ConvertScalarFieldValue(value, field, '{0}.{1}'.format(path, name)))
        except ParseError as e:
            if field and field.containing_oneof is None:
                raise ParseError('Failed to parse {0} field: {1}.'.format(name, e)) from e
            else:
                raise ParseError(str(e)) from e
        except ValueError as e:
            raise ParseError('Failed to parse {0} field: {1}.'.format(name, e)) from e
        except TypeError as e:
            raise ParseError('Failed to parse {0} field: {1}.'.format(name, e)) from e