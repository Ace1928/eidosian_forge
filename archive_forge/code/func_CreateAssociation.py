from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def CreateAssociation(self, parent, network, firewall_endpoint, association_id=None, tls_inspection_policy=None, labels=None):
    """Calls the CreateAssociation API.

    Args:
      parent: The parent of the association, e.g.
        "projects/myproj/locations/us-central1-a"
      network: The network of the association, e.g.
        "projects/myproj/networks/global/my-vpc"
      firewall_endpoint: The firewall endpoint of the association, e.g. "
        organizations/123456/locations/us-central1-a/firewallEndpoints/my-ep"
      association_id: The ID of the association, e.g. "my-assoc".
      tls_inspection_policy: The TLS inspection policy of the association, e.g.
        "projects/my-proj/locations/us-central1/tlsInspectionPolicies/my-tls".
      labels: A dictionary with the labels of the association.

    Returns:
      NetworksecurityProjectsLocationsFirewallEndpointAssociationsCreateResponse
    """
    association = self.messages.FirewallEndpointAssociation(network=network, firewallEndpoint=firewall_endpoint, labels=labels, tlsInspectionPolicy=tls_inspection_policy)
    create_request = self.messages.NetworksecurityProjectsLocationsFirewallEndpointAssociationsCreateRequest(firewallEndpointAssociation=association, firewallEndpointAssociationId=association_id, parent=parent)
    return self._association_client.Create(create_request)