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
def _PrintMessageFieldValue(self, value):
    if self.pointy_brackets:
        openb = '<'
        closeb = '>'
    else:
        openb = '{'
        closeb = '}'
    if self.as_one_line:
        self.out.write('%s ' % openb)
        self.PrintMessage(value)
        self.out.write(closeb)
    else:
        self.out.write('%s\n' % openb)
        self.indent += 2
        self.PrintMessage(value)
        self.indent -= 2
        self.out.write(' ' * self.indent + closeb)