from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartnerSSEGatewayPartnerSSEGatewaySymantecOptions(_messages.Message):
    """Options specific to gateways connected to Symantec.

  Fields:
    symantecLocationUuid: Output only. UUID of the Symantec Location created
      on the customer's behalf.
    symantecSite: Output only. Symantec data center identifier that this SSEGW
      will connect to. Filled from the customer SSEGateway, and only for
      PartnerSSEGateways associated with Symantec today.
    symantecSiteTargetHost: Optional. Target for the NCGs to send traffic to
      on the Symantec side. Only supports IP address today.
  """
    symantecLocationUuid = _messages.StringField(1)
    symantecSite = _messages.StringField(2)
    symantecSiteTargetHost = _messages.StringField(3)