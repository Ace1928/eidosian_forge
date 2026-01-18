from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.util.apis import arg_utils
def _DetachZonalEndpoints(self, neg_ref, endpoints):
    """Detaches network endpoints from a zonal network endpoint group."""
    request_class = self.messages.ComputeNetworkEndpointGroupsDetachNetworkEndpointsRequest
    nested_request_class = self.messages.NetworkEndpointGroupsDetachEndpointsRequest
    request = request_class(networkEndpointGroup=neg_ref.Name(), project=neg_ref.project, zone=neg_ref.zone, networkEndpointGroupsDetachEndpointsRequest=nested_request_class(networkEndpoints=self._GetEndpointMessageList(endpoints)))
    return self._zonal_service.DetachNetworkEndpoints(request)