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
class TextWriter(object):

    def __init__(self, as_utf8):
        self._writer = io.StringIO()

    def write(self, val):
        return self._writer.write(val)

    def close(self):
        return self._writer.close()

    def getvalue(self):
        return self._writer.getvalue()