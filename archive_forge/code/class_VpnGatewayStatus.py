from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpnGatewayStatus(_messages.Message):
    """A VpnGatewayStatus object.

  Fields:
    vpnConnections: List of VPN connection for this VpnGateway.
  """
    vpnConnections = _messages.MessageField('VpnGatewayStatusVpnConnection', 1, repeated=True)