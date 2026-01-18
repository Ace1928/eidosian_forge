from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
def _MakeListAssociationsRequestTuple(self, target_resource):
    return (self._client.firewallPolicies, 'ListAssociations', self._messages.ComputeFirewallPoliciesListAssociationsRequest(targetResource=target_resource))