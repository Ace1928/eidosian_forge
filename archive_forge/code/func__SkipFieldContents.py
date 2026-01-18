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
def _SkipFieldContents(self, tokenizer, field_name, immediate_message_type):
    """Skips over contents (value or message) of a field.

    Args:
      tokenizer: A tokenizer to parse the field name and values.
      field_name: The field name currently being parsed.
      immediate_message_type: The type of the message immediately containing
        the silent marker.
    """
    if tokenizer.TryConsume(':') and (not tokenizer.LookingAt('{')) and (not tokenizer.LookingAt('<')):
        self._DetectSilentMarker(tokenizer, immediate_message_type, field_name)
        if tokenizer.LookingAt('['):
            self._SkipRepeatedFieldValue(tokenizer)
        else:
            self._SkipFieldValue(tokenizer)
    else:
        self._DetectSilentMarker(tokenizer, immediate_message_type, field_name)
        self._SkipFieldMessage(tokenizer, immediate_message_type)