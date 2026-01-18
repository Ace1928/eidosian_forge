from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HarmTypeValueListEntryValuesEnum(_messages.Enum):
    """HarmTypeValueListEntryValuesEnum enum type.

    Values:
      HARM_TYPE_UNSPECIFIED: <no description>
      HARM_TYPE_HATE: <no description>
      HARM_TYPE_TOXICITY: <no description>
      HARM_TYPE_VIOLENCE: <no description>
      HARM_TYPE_CSAI: <no description>
      HARM_TYPE_SEXUAL: <no description>
      HARM_TYPE_FRINGE: <no description>
      HARM_TYPE_POLITICAL: <no description>
      HARM_TYPE_MEMORIZATION: <no description>
      HARM_TYPE_SPII: <no description>
      HARM_TYPE_NEW_DANGEROUS: New definition of dangerous.
      HARM_TYPE_MEDICAL: <no description>
      HARM_TYPE_HARASSMENT: <no description>
    """
    HARM_TYPE_UNSPECIFIED = 0
    HARM_TYPE_HATE = 1
    HARM_TYPE_TOXICITY = 2
    HARM_TYPE_VIOLENCE = 3
    HARM_TYPE_CSAI = 4
    HARM_TYPE_SEXUAL = 5
    HARM_TYPE_FRINGE = 6
    HARM_TYPE_POLITICAL = 7
    HARM_TYPE_MEMORIZATION = 8
    HARM_TYPE_SPII = 9
    HARM_TYPE_NEW_DANGEROUS = 10
    HARM_TYPE_MEDICAL = 11
    HARM_TYPE_HARASSMENT = 12