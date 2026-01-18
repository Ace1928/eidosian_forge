from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
def GetHighAvailabilityVpnTunnelForInsert(self, name, description, ike_version, peer_ip, shared_secret, vpn_gateway, vpn_gateway_interface, router, peer_external_gateway, peer_external_gateway_interface, peer_gcp_gateway):
    """Returns the HA VpnTunnel message for an insert request.

    Args:
      name: String representing the name of the VPN tunnel resource.
      description: String representing the description for the VPN tunnel
        resource.
      ike_version: The IKE protocol version for establishing the VPN tunnel.
      peer_ip: String representing the peer IP address for the VPN tunnel.
      shared_secret: String representing the shared secret (IKE pre-shared key).
      vpn_gateway: String representing the VPN Gateway URL the VPN tunnel
        resource should be associated with.
      vpn_gateway_interface: Integer representing the VPN Gateway interface ID
        that VPN tunnel resource should be associated with.
      router: String representing the Router URL the VPN tunnel resource should
        be associated with.
      peer_external_gateway: String representing of the peer side external VPN
        gateway to which the VPN tunnel is connected.
      peer_external_gateway_interface: Interface ID of the External VPN gateway
        to which this VPN tunnel is connected.
      peer_gcp_gateway:  String representing of peer side HA GCP VPN gateway
        to which this VPN tunnel is connected.

    Returns:
      The VpnTunnel message object that can be used in an insert request.
    """
    return self._messages.VpnTunnel(name=name, description=description, ikeVersion=ike_version, peerIp=peer_ip, sharedSecret=shared_secret, vpnGateway=vpn_gateway, vpnGatewayInterface=vpn_gateway_interface, router=router, peerExternalGateway=peer_external_gateway, peerExternalGatewayInterface=peer_external_gateway_interface, peerGcpGateway=peer_gcp_gateway)