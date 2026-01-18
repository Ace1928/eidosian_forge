from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.batch import util as batch_api_util
class ResourceAllowancesClient(object):
    """Client for resource allowances service in the Cloud Batch API."""

    def __init__(self, release_track, client=None, messages=None):
        self.client = client or batch_api_util.GetClientInstance(release_track)
        self.messages = messages or self.client.MESSAGES_MODULE
        self.service = self.client.projects_locations_resourceAllowances

    def Create(self, resource_allowance_id, location_ref, resource_allowance_config):
        create_req_type = self.messages.BatchProjectsLocationsResourceAllowancesCreateRequest
        create_req = create_req_type(resourceAllowanceId=resource_allowance_id, parent=location_ref.RelativeName(), resourceAllowance=resource_allowance_config)
        return self.service.Create(create_req)

    def Get(self, resource_allowance_ref):
        get_req_type = self.messages.BatchProjectsLocationsResourceAllowancesGetRequest
        get_req = get_req_type(name=resource_allowance_ref.RelativeName())
        return self.service.Get(get_req)

    def Delete(self, resource_allowance_ref):
        delete_req_type = self.messages.BatchProjectsLocationsResourceAllowancesDeleteRequest
        delete_req = delete_req_type(name=resource_allowance_ref.RelativeName())
        return self.service.Delete(delete_req)

    def Update(self, resource_allowance_ref, resource_allowance_config, update_mask):
        update_req_type = self.messages.BatchProjectsLocationsResourceAllowancesPatchRequest
        update_req = update_req_type(name=resource_allowance_ref.RelativeName(), updateMask=','.join(update_mask), resourceAllowance=resource_allowance_config)
        return self.service.Patch(update_req)