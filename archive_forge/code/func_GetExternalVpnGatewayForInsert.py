from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
def GetExternalVpnGatewayForInsert(self, name, description, redundancy_type, interfaces):
    """Returns the VpnGateway message for an insert request.

    Args:
      name: String representing the name of the external VPN Gateway resource.
      description: String representing the description for the VPN Gateway
        resource.
      redundancy_type: Redundancy type of the external VPN gateway.
      interfaces: list of interfaces for the external VPN gateway

    Returns:
      The ExternalVpnGateway message object that can be used in an insert
      request.
    """
    return self._messages.ExternalVpnGateway(name=name, description=description, redundancyType=redundancy_type, interfaces=interfaces)