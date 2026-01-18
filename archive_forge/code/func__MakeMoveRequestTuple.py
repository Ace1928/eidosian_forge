from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
def _MakeMoveRequestTuple(self, fp_id=None, parent_id=None):
    return (self._client.firewallPolicies, 'Move', self._messages.ComputeFirewallPoliciesMoveRequest(firewallPolicy=fp_id, parentId=parent_id))