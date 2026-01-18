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
def _SkipFieldMessage(self, tokenizer, immediate_message_type):
    """Skips over a field message.

    Args:
      tokenizer: A tokenizer to parse the field name and values.
      immediate_message_type: The type of the message immediately containing
        the silent marker
    """
    if tokenizer.TryConsume('<'):
        delimiter = '>'
    else:
        tokenizer.Consume('{')
        delimiter = '}'
    while not tokenizer.LookingAt('>') and (not tokenizer.LookingAt('}')):
        self._SkipField(tokenizer, immediate_message_type)
    tokenizer.Consume(delimiter)