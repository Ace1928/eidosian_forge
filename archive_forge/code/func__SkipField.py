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
def _SkipField(self, tokenizer, immediate_message_type):
    """Skips over a complete field (name and value/message).

    Args:
      tokenizer: A tokenizer to parse the field name and values.
      immediate_message_type: The type of the message immediately containing
        the silent marker.
    """
    field_name = ''
    if tokenizer.TryConsume('['):
        field_name += '[' + tokenizer.ConsumeIdentifier()
        num_identifiers = 1
        while tokenizer.TryConsume('.'):
            field_name += '.' + tokenizer.ConsumeIdentifier()
            num_identifiers += 1
        if num_identifiers == 3 and tokenizer.TryConsume('/'):
            field_name += '/' + tokenizer.ConsumeIdentifier()
            while tokenizer.TryConsume('.'):
                field_name += '.' + tokenizer.ConsumeIdentifier()
        tokenizer.Consume(']')
        field_name += ']'
    else:
        field_name += tokenizer.ConsumeIdentifierOrNumber()
    self._SkipFieldContents(tokenizer, field_name, immediate_message_type)
    if not tokenizer.TryConsume(','):
        tokenizer.TryConsume(';')