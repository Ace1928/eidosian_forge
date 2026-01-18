from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.network_connectivity import networkconnectivity_util
from googlecloudsdk.calliope import base
def CreateVPCSpoke(self, spoke_ref, spoke, request_id=None):
    """Call API to create a new spoke."""
    parent = spoke_ref.Parent().RelativeName()
    spoke_id = spoke_ref.Name()
    create_req = self.messages.NetworkconnectivityProjectsLocationsSpokesCreateRequest(parent=parent, requestId=request_id, spoke=spoke, spokeId=spoke_id)
    return self.spoke_service.Create(create_req)