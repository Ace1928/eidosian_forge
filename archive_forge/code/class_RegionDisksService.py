from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionDisksService(base_api.BaseApiService):
    """Service class for the regionDisks resource."""
    _NAME = 'regionDisks'

    def __init__(self, client):
        super(ComputeBeta.RegionDisksService, self).__init__(client)
        self._upload_configs = {}

    def AddResourcePolicies(self, request, global_params=None):
        """Adds existing resource policies to a regional disk. You can only add one policy which will be applied to this disk for scheduling snapshot creation.

      Args:
        request: (ComputeRegionDisksAddResourcePoliciesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddResourcePolicies')
        return self._RunMethod(config, request, global_params=global_params)
    AddResourcePolicies.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionDisks.addResourcePolicies', ordered_params=['project', 'region', 'disk'], path_params=['disk', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/disks/{disk}/addResourcePolicies', request_field='regionDisksAddResourcePoliciesRequest', request_type_name='ComputeRegionDisksAddResourcePoliciesRequest', response_type_name='Operation', supports_download=False)

    def BulkInsert(self, request, global_params=None):
        """Bulk create a set of disks.

      Args:
        request: (ComputeRegionDisksBulkInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('BulkInsert')
        return self._RunMethod(config, request, global_params=global_params)
    BulkInsert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionDisks.bulkInsert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/disks/bulkInsert', request_field='bulkInsertDiskResource', request_type_name='ComputeRegionDisksBulkInsertRequest', response_type_name='Operation', supports_download=False)

    def CreateSnapshot(self, request, global_params=None):
        """Creates a snapshot of a specified persistent disk. For regular snapshot creation, consider using snapshots.insert instead, as that method supports more features, such as creating snapshots in a project different from the source disk project.

      Args:
        request: (ComputeRegionDisksCreateSnapshotRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('CreateSnapshot')
        return self._RunMethod(config, request, global_params=global_params)
    CreateSnapshot.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionDisks.createSnapshot', ordered_params=['project', 'region', 'disk'], path_params=['disk', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/disks/{disk}/createSnapshot', request_field='snapshot', request_type_name='ComputeRegionDisksCreateSnapshotRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified regional persistent disk. Deleting a regional disk removes all the replicas of its data permanently and is irreversible. However, deleting a disk does not delete any snapshots previously made from the disk. You must separately delete snapshots.

      Args:
        request: (ComputeRegionDisksDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.regionDisks.delete', ordered_params=['project', 'region', 'disk'], path_params=['disk', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/disks/{disk}', request_field='', request_type_name='ComputeRegionDisksDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns a specified regional persistent disk.

      Args:
        request: (ComputeRegionDisksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Disk) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionDisks.get', ordered_params=['project', 'region', 'disk'], path_params=['disk', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/disks/{disk}', request_field='', request_type_name='ComputeRegionDisksGetRequest', response_type_name='Disk', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeRegionDisksGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionDisks.getIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/regions/{region}/disks/{resource}/getIamPolicy', request_field='', request_type_name='ComputeRegionDisksGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a persistent regional disk in the specified project using the data included in the request.

      Args:
        request: (ComputeRegionDisksInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionDisks.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId', 'sourceImage'], relative_path='projects/{project}/regions/{region}/disks', request_field='disk', request_type_name='ComputeRegionDisksInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of persistent disks contained within the specified region.

      Args:
        request: (ComputeRegionDisksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DiskList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionDisks.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/disks', request_field='', request_type_name='ComputeRegionDisksListRequest', response_type_name='DiskList', supports_download=False)

    def RemoveResourcePolicies(self, request, global_params=None):
        """Removes resource policies from a regional disk.

      Args:
        request: (ComputeRegionDisksRemoveResourcePoliciesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RemoveResourcePolicies')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveResourcePolicies.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionDisks.removeResourcePolicies', ordered_params=['project', 'region', 'disk'], path_params=['disk', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/disks/{disk}/removeResourcePolicies', request_field='regionDisksRemoveResourcePoliciesRequest', request_type_name='ComputeRegionDisksRemoveResourcePoliciesRequest', response_type_name='Operation', supports_download=False)

    def Resize(self, request, global_params=None):
        """Resizes the specified regional persistent disk.

      Args:
        request: (ComputeRegionDisksResizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Resize')
        return self._RunMethod(config, request, global_params=global_params)
    Resize.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionDisks.resize', ordered_params=['project', 'region', 'disk'], path_params=['disk', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/disks/{disk}/resize', request_field='regionDisksResizeRequest', request_type_name='ComputeRegionDisksResizeRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeRegionDisksSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionDisks.setIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/disks/{resource}/setIamPolicy', request_field='regionSetPolicyRequest', request_type_name='ComputeRegionDisksSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def SetLabels(self, request, global_params=None):
        """Sets the labels on the target regional disk.

      Args:
        request: (ComputeRegionDisksSetLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLabels')
        return self._RunMethod(config, request, global_params=global_params)
    SetLabels.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionDisks.setLabels', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/disks/{resource}/setLabels', request_field='regionSetLabelsRequest', request_type_name='ComputeRegionDisksSetLabelsRequest', response_type_name='Operation', supports_download=False)

    def StartAsyncReplication(self, request, global_params=None):
        """Starts asynchronous replication. Must be invoked on the primary disk.

      Args:
        request: (ComputeRegionDisksStartAsyncReplicationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('StartAsyncReplication')
        return self._RunMethod(config, request, global_params=global_params)
    StartAsyncReplication.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionDisks.startAsyncReplication', ordered_params=['project', 'region', 'disk'], path_params=['disk', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/disks/{disk}/startAsyncReplication', request_field='regionDisksStartAsyncReplicationRequest', request_type_name='ComputeRegionDisksStartAsyncReplicationRequest', response_type_name='Operation', supports_download=False)

    def StopAsyncReplication(self, request, global_params=None):
        """Stops asynchronous replication. Can be invoked either on the primary or on the secondary disk.

      Args:
        request: (ComputeRegionDisksStopAsyncReplicationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('StopAsyncReplication')
        return self._RunMethod(config, request, global_params=global_params)
    StopAsyncReplication.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionDisks.stopAsyncReplication', ordered_params=['project', 'region', 'disk'], path_params=['disk', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/disks/{disk}/stopAsyncReplication', request_field='', request_type_name='ComputeRegionDisksStopAsyncReplicationRequest', response_type_name='Operation', supports_download=False)

    def StopGroupAsyncReplication(self, request, global_params=None):
        """Stops asynchronous replication for a consistency group of disks. Can be invoked either in the primary or secondary scope.

      Args:
        request: (ComputeRegionDisksStopGroupAsyncReplicationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('StopGroupAsyncReplication')
        return self._RunMethod(config, request, global_params=global_params)
    StopGroupAsyncReplication.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionDisks.stopGroupAsyncReplication', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/disks/stopGroupAsyncReplication', request_field='disksStopGroupAsyncReplicationResource', request_type_name='ComputeRegionDisksStopGroupAsyncReplicationRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeRegionDisksTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionDisks.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/disks/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRegionDisksTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Update the specified disk with the data included in the request. Update is performed only on selected fields included as part of update-mask. Only the following fields can be modified: user_license.

      Args:
        request: (ComputeRegionDisksUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.regionDisks.update', ordered_params=['project', 'region', 'disk'], path_params=['disk', 'project', 'region'], query_params=['paths', 'requestId', 'updateMask'], relative_path='projects/{project}/regions/{region}/disks/{disk}', request_field='diskResource', request_type_name='ComputeRegionDisksUpdateRequest', response_type_name='Operation', supports_download=False)