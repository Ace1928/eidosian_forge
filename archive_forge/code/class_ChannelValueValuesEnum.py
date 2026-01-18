from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChannelValueValuesEnum(_messages.Enum):
    """Release Channel the managed control plane revision is subscribed to.

    Values:
      CHANNEL_UNSPECIFIED: Unspecified
      RAPID: RAPID channel is offered on an early access basis for customers
        who want to test new releases.
      REGULAR: REGULAR channel is intended for production users who want to
        take advantage of new features.
      STABLE: STABLE channel includes versions that are known to be stable and
        reliable in production.
    """
    CHANNEL_UNSPECIFIED = 0
    RAPID = 1
    REGULAR = 2
    STABLE = 3