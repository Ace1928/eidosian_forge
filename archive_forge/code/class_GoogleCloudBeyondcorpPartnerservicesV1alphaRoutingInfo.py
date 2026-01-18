from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpPartnerservicesV1alphaRoutingInfo(_messages.Message):
    """Message contains the routing information to direct traffic to the proxy
  server.

  Fields:
    pacUri: Required. Proxy Auto-Configuration (PAC) URI.
  """
    pacUri = _messages.StringField(1)