from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyValueValuesEnum(_messages.Enum):
    """PolicyValueValuesEnum enum type.

    Values:
      UNSPECIFIED: <no description>
      DANGEROUS_CONTENT: <no description>
      HARASSMENT: <no description>
      HATE_SPEECH: <no description>
      SEXUALLY_EXPLICIT: <no description>
    """
    UNSPECIFIED = 0
    DANGEROUS_CONTENT = 1
    HARASSMENT = 2
    HATE_SPEECH = 3
    SEXUALLY_EXPLICIT = 4