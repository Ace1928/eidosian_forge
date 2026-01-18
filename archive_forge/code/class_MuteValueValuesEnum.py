from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MuteValueValuesEnum(_messages.Enum):
    """Required. The desired state of the Mute.

    Values:
      MUTE_UNSPECIFIED: Unspecified.
      MUTED: Finding has been muted.
      UNMUTED: Finding has been unmuted.
      UNDEFINED: Finding has never been muted/unmuted.
    """
    MUTE_UNSPECIFIED = 0
    MUTED = 1
    UNMUTED = 2
    UNDEFINED = 3