from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeVpnGatewaysGetRequest(_messages.Message):
    """A ComputeVpnGatewaysGetRequest object.

  Fields:
    project: Project ID for this request.
    region: Name of the region for this request.
    vpnGateway: Name of the VPN gateway to return.
  """
    project = _messages.StringField(1, required=True)
    region = _messages.StringField(2, required=True)
    vpnGateway = _messages.StringField(3, required=True)