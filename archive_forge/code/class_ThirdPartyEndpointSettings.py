from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ThirdPartyEndpointSettings(_messages.Message):
    """Next ID: 2.

  Fields:
    targetFirewallAttachment: Optional. URL of the target firewall attachment.
  """
    targetFirewallAttachment = _messages.StringField(1)