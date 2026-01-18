from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChannelTypeValueValuesEnum(_messages.Enum):
    """Type of channel used.

    Values:
      CHANNEL_TYPE_UNSPECIFIED: <no description>
      CHANNEL_TYPE_GAA: <no description>
      CHANNEL_TYPE_PAL: <no description>
    """
    CHANNEL_TYPE_UNSPECIFIED = 0
    CHANNEL_TYPE_GAA = 1
    CHANNEL_TYPE_PAL = 2