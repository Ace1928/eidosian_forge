from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AttachZonalEndpoints(self, neg_ref, endpoints):
    """Attaches network endpoints to a zonal network endpoint group."""
    request_class = self.messages.ComputeNetworkEndpointGroupsAttachNetworkEndpointsRequest
    nested_request_class = self.messages.NetworkEndpointGroupsAttachEndpointsRequest
    request = request_class(networkEndpointGroup=neg_ref.Name(), project=neg_ref.project, zone=neg_ref.zone, networkEndpointGroupsAttachEndpointsRequest=nested_request_class(networkEndpoints=self._GetEndpointMessageList(endpoints)))
    return self._zonal_service.AttachNetworkEndpoints(request)