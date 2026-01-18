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
def MessageToBytes(message, **kwargs) -> bytes:
    """Convert protobuf message to encoded text format.  See MessageToString."""
    text = MessageToString(message, **kwargs)
    if isinstance(text, bytes):
        return text
    codec = 'utf-8' if kwargs.get('as_utf8') else 'ascii'
    return text.encode(codec)