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
def _SkipRepeatedFieldValue(self, tokenizer):
    """Skips over a repeated field value.

    Args:
      tokenizer: A tokenizer to parse the field value.
    """
    tokenizer.Consume('[')
    if not tokenizer.LookingAt(']'):
        self._SkipFieldValue(tokenizer)
        while tokenizer.TryConsume(','):
            self._SkipFieldValue(tokenizer)
    tokenizer.Consume(']')