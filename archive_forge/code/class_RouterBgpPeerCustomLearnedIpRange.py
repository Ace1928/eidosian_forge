from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouterBgpPeerCustomLearnedIpRange(_messages.Message):
    """A RouterBgpPeerCustomLearnedIpRange object.

  Fields:
    range: The custom learned route IP address range. Must be a valid CIDR-
      formatted prefix. If an IP address is provided without a subnet mask, it
      is interpreted as, for IPv4, a `/32` singular IP address range, and, for
      IPv6, `/128`.
  """
    range = _messages.StringField(1)