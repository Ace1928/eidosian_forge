from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.network_connectivity import networkconnectivity_util
from googlecloudsdk.calliope import base
class GroupsClient(object):
    """Client for group service in network connectivity API."""

    def __init__(self, release_track=base.ReleaseTrack.GA):
        self.release_track = release_track
        self.client = networkconnectivity_util.GetClientInstance(release_track)
        self.messages = networkconnectivity_util.GetMessagesModule(release_track)
        self.group_service = self.client.projects_locations_global_hubs_groups
        self.operation_service = self.client.projects_locations_operations

    def UpdateGroup(self, group_ref, group, update_mask, request_id=None):
        """Call API to update an existing group."""
        name = group_ref.RelativeName()
        update_mask_string = ','.join(update_mask)
        update_req = self.messages.NetworkconnectivityProjectsLocationsGlobalHubsGroupsPatchRequest(name=name, requestId=request_id, group=group, updateMask=update_mask_string)
        return self.group_service.Patch(update_req)

    def Get(self, group_ref):
        """Call API to get an existing group."""
        get_req = self.messages.NetworkconnectivityProjectsLocationsGlobalHubsGroupsGetRequest(name=group_ref.RelativeName())
        return self.group_service.Get(get_req)