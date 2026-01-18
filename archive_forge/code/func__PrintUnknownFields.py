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
def _PrintUnknownFields(self, unknown_field_set):
    """Print unknown fields."""
    out = self.out
    for field in unknown_field_set:
        out.write(' ' * self.indent)
        out.write(str(field.field_number))
        if field.wire_type == WIRETYPE_START_GROUP:
            if self.as_one_line:
                out.write(' { ')
            else:
                out.write(' {\n')
                self.indent += 2
            self._PrintUnknownFields(field.data)
            if self.as_one_line:
                out.write('} ')
            else:
                self.indent -= 2
                out.write(' ' * self.indent + '}\n')
        elif field.wire_type == WIRETYPE_LENGTH_DELIMITED:
            try:
                embedded_unknown_message, pos = decoder._DecodeUnknownFieldSet(memoryview(field.data), 0, len(field.data))
            except Exception:
                pos = 0
            if pos == len(field.data):
                if self.as_one_line:
                    out.write(' { ')
                else:
                    out.write(' {\n')
                    self.indent += 2
                self._PrintUnknownFields(embedded_unknown_message)
                if self.as_one_line:
                    out.write('} ')
                else:
                    self.indent -= 2
                    out.write(' ' * self.indent + '}\n')
            else:
                out.write(': "')
                out.write(text_encoding.CEscape(field.data, False))
                out.write('" ' if self.as_one_line else '"\n')
        else:
            out.write(': ')
            out.write(str(field.data))
            out.write(' ' if self.as_one_line else '\n')