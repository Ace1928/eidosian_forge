from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchematizedParsingTypeValueValuesEnum(_messages.Enum):
    """Determines how messages that fail to parse are handled.

    Values:
      SCHEMATIZED_PARSING_TYPE_UNSPECIFIED: Unspecified schematized parsing
        type, equivalent to `SOFT_FAIL`.
      SOFT_FAIL: Messages that fail to parse are still stored and ACKed but a
        parser error is stored in place of the schematized data.
      HARD_FAIL: Messages that fail to parse are rejected from
        ingestion/insertion and return an error code.
    """
    SCHEMATIZED_PARSING_TYPE_UNSPECIFIED = 0
    SOFT_FAIL = 1
    HARD_FAIL = 2