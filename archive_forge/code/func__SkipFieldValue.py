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
def _SkipFieldValue(self, tokenizer):
    """Skips over a field value.

    Args:
      tokenizer: A tokenizer to parse the field name and values.

    Raises:
      ParseError: In case an invalid field value is found.
    """
    if not tokenizer.TryConsumeByteString() and (not tokenizer.TryConsumeIdentifier()) and (not _TryConsumeInt64(tokenizer)) and (not _TryConsumeUint64(tokenizer)) and (not tokenizer.TryConsumeFloat()):
        raise ParseError('Invalid field value: ' + tokenizer.token)