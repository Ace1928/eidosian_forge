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
def Consume(self, token):
    """Consumes a piece of text.

    Args:
      token: Text to consume.

    Raises:
      ParseError: If the text couldn't be consumed.
    """
    if not self.TryConsume(token):
        raise self.ParseError('Expected "%s".' % token)