from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpnGatewayInfo(_messages.Message):
    """For display only. Metadata associated with a Compute Engine VPN gateway.

  Fields:
    displayName: Name of a VPN gateway.
    ipAddress: IP address of the VPN gateway.
    networkUri: URI of a Compute Engine network where the VPN gateway is
      configured.
    region: Name of a Google Cloud region where this VPN gateway is
      configured.
    uri: URI of a VPN gateway.
    vpnTunnelUri: A VPN tunnel that is associated with this VPN gateway. There
      may be multiple VPN tunnels configured on a VPN gateway, and only the
      one relevant to the test is displayed.
  """
    displayName = _messages.StringField(1)
    ipAddress = _messages.StringField(2)
    networkUri = _messages.StringField(3)
    region = _messages.StringField(4)
    uri = _messages.StringField(5)
    vpnTunnelUri = _messages.StringField(6)