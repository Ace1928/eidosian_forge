from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudVpn(_messages.Message):
    """The Cloud VPN info.

  Fields:
    gateway: The created Cloud VPN gateway name.
  """
    gateway = _messages.StringField(1)