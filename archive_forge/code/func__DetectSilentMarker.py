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
def _DetectSilentMarker(self, tokenizer, immediate_message_type, field_name):
    if tokenizer.contains_silent_marker_before_current_token:
        self._LogSilentMarker(immediate_message_type, field_name)