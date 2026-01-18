import encodings.raw_unicode_escape  # pylint: disable=unused-import
import encodings.unicode_escape  # pylint: disable=unused-import
import io
import math
import re
from google.protobuf.internal import decoder
from google.protobuf.internal import type_checkers
from google.protobuf import descriptor
from google.protobuf import text_encoding
from google.protobuf import unknown_fields
def PrintFieldValue(self, field, value):
    """Print a single field value (not including name).

    For repeated fields, the value should be a single element.

    Args:
      field: The descriptor of the field to be printed.
      value: The value of the field.
    """
    out = self.out
    if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
        self._PrintMessageFieldValue(value)
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_ENUM:
        enum_value = field.enum_type.values_by_number.get(value, None)
        if enum_value is not None:
            out.write(enum_value.name)
        else:
            out.write(str(value))
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_STRING:
        out.write('"')
        if isinstance(value, str) and (not self.as_utf8):
            out_value = value.encode('utf-8')
        else:
            out_value = value
        if field.type == descriptor.FieldDescriptor.TYPE_BYTES:
            out_as_utf8 = False
        else:
            out_as_utf8 = self.as_utf8
        out.write(text_encoding.CEscape(out_value, out_as_utf8))
        out.write('"')
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_BOOL:
        if value:
            out.write('true')
        else:
            out.write('false')
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_FLOAT:
        if self.float_format is not None:
            out.write('{1:{0}}'.format(self.float_format, value))
        elif math.isnan(value):
            out.write(str(value))
        else:
            out.write(str(type_checkers.ToShortestFloat(value)))
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_DOUBLE and self.double_format is not None:
        out.write('{1:{0}}'.format(self.double_format, value))
    else:
        out.write(str(value))