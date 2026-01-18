from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UnexpectedSegmentHandlingValueValuesEnum(_messages.Enum):
    """Determines how unexpected segments (segments not matched to the
    schema) are handled.

    Values:
      UNEXPECTED_SEGMENT_HANDLING_MODE_UNSPECIFIED: Unspecified handling mode,
        equivalent to FAIL.
      FAIL: Unexpected segments fail to parse and return an error.
      SKIP: Unexpected segments do not fail, but are omitted from the output.
      PARSE: Unexpected segments do not fail, but are parsed in place and
        added to the current group. If a segment has a type definition, it is
        used, otherwise it is parsed as VARIES.
    """
    UNEXPECTED_SEGMENT_HANDLING_MODE_UNSPECIFIED = 0
    FAIL = 1
    SKIP = 2
    PARSE = 3