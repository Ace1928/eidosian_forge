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
def ConsumeEnum(self, field):
    try:
        result = ParseEnum(field, self.token)
    except ValueError as e:
        raise self.ParseError(str(e))
    self.NextToken()
    return result