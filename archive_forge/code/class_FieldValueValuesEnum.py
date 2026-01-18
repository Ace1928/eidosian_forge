from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FieldValueValuesEnum(_messages.Enum):
    """The field that is set in the API proto.

    Values:
      IDENTIFIER_HELPER_FIELD_UNSPECIFIED: The helper isn't set.
      GENERIC_URI: The generic_uri one-of field is set.
    """
    IDENTIFIER_HELPER_FIELD_UNSPECIFIED = 0
    GENERIC_URI = 1