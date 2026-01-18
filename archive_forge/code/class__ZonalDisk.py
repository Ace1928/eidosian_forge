from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.core.exceptions import Error
class _ZonalDisk(object):
    """A wrapper for Compute Engine DisksService API client."""

    def __init__(self, client, disk_ref, messages):
        self._disk_ref = disk_ref
        self._client = client
        self._service = client.disks or client.apitools_client.disks
        self._messages = messages

    @classmethod
    def GetOperationCollection(cls):
        """Gets the zonal operation collection of a compute disk reference."""
        return 'compute.zoneOperations'

    def GetService(self):
        return self._service

    def GetDiskRequestMessage(self):
        """Gets the zonal compute disk get request message."""
        return self._messages.ComputeDisksGetRequest(**self._disk_ref.AsDict())

    def GetDiskResource(self):
        request_msg = self.GetDiskRequestMessage()
        return self._service.Get(request_msg)

    def GetSetLabelsRequestMessage(self):
        return self._messages.ZoneSetLabelsRequest

    def GetSetDiskLabelsRequestMessage(self, disk, labels):
        req = self._messages.ComputeDisksSetLabelsRequest
        return req(project=self._disk_ref.project, resource=self._disk_ref.disk, zone=self._disk_ref.zone, zoneSetLabelsRequest=self._messages.ZoneSetLabelsRequest(labelFingerprint=disk.labelFingerprint, labels=labels))

    def GetDiskRegionName(self):
        return compute_utils.ZoneNameToRegionName(self._disk_ref.zone)

    def MakeAddResourcePoliciesRequest(self, resource_policies, client_to_make_request):
        add_request = self._messages.ComputeDisksAddResourcePoliciesRequest(disk=self._disk_ref.Name(), project=self._disk_ref.project, zone=self._disk_ref.zone, disksAddResourcePoliciesRequest=self._messages.DisksAddResourcePoliciesRequest(resourcePolicies=resource_policies))
        return client_to_make_request.MakeRequests([(self._client.disks, 'AddResourcePolicies', add_request)])

    def MakeRemoveResourcePoliciesRequest(self, resource_policies, client_to_make_request):
        remove_request = self._messages.ComputeDisksRemoveResourcePoliciesRequest(disk=self._disk_ref.Name(), project=self._disk_ref.project, zone=self._disk_ref.zone, disksRemoveResourcePoliciesRequest=self._messages.DisksRemoveResourcePoliciesRequest(resourcePolicies=resource_policies))
        return client_to_make_request.MakeRequests([(self._client.disks, 'RemoveResourcePolicies', remove_request)])