from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JsonParsingValueValuesEnum(_messages.Enum):
    """JsonParsingValueValuesEnum enum type.

    Values:
      DISABLED: <no description>
      STANDARD: <no description>
      STANDARD_WITH_GRAPHQL: <no description>
    """
    DISABLED = 0
    STANDARD = 1
    STANDARD_WITH_GRAPHQL = 2