from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecondaryIpRange(_messages.Message):
    """A SecondaryIpRange object.

  Fields:
    ipCidrRange: Secondary IP CIDR range in `x.x.x.x/y` format.
    rangeName: Name of the secondary IP range.
  """
    ipCidrRange = _messages.StringField(1)
    rangeName = _messages.StringField(2)