import encodings.raw_unicode_escape  # pylint: disable=unused-import
import encodings.unicode_escape  # pylint: disable=unused-import
import io
import math
import re
from cloudsdk.google.protobuf.internal import decoder
from cloudsdk.google.protobuf.internal import type_checkers
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import text_encoding
@staticmethod
def _IsProto3Syntax(message):
    message_descriptor = message.DESCRIPTOR
    return hasattr(message_descriptor, 'syntax') and message_descriptor.syntax == 'proto3'