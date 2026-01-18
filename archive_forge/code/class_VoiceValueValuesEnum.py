from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VoiceValueValuesEnum(_messages.Enum):
    """The grammatical voice.

    Values:
      VOICE_UNKNOWN: Voice is not applicable in the analyzed language or is
        not predicted.
      ACTIVE: Active
      CAUSATIVE: Causative
      PASSIVE: Passive
    """
    VOICE_UNKNOWN = 0
    ACTIVE = 1
    CAUSATIVE = 2
    PASSIVE = 3