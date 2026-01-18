from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.util.apis import arg_utils
def _DetachGlobalEndpoints(self, neg_ref, endpoints):
    """Detaches network endpoints from a global network endpoint group."""
    request_class = self.messages.ComputeGlobalNetworkEndpointGroupsDetachNetworkEndpointsRequest
    nested_request_class = self.messages.GlobalNetworkEndpointGroupsDetachEndpointsRequest
    request = request_class(networkEndpointGroup=neg_ref.Name(), project=neg_ref.project, globalNetworkEndpointGroupsDetachEndpointsRequest=nested_request_class(networkEndpoints=self._GetEndpointMessageList(endpoints)))
    return self._global_service.DetachNetworkEndpoints(request)