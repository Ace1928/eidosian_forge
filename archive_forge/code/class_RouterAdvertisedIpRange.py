from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouterAdvertisedIpRange(_messages.Message):
    """Description-tagged IP ranges for the router to advertise.

  Fields:
    description: User-specified description for the IP range.
    range: The IP range to advertise. The value must be a CIDR-formatted
      string.
  """
    description = _messages.StringField(1)
    range = _messages.StringField(2)