from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class DisksService(base_api.BaseApiService):
    """Service class for the disks resource."""
    _NAME = 'disks'

    def __init__(self, client):
        super(ComputeBeta.DisksService, self).__init__(client)
        self._upload_configs = {}

    def AddResourcePolicies(self, request, global_params=None):
        """Adds existing resource policies to a disk. You can only add one policy which will be applied to this disk for scheduling snapshot creation.

      Args:
        request: (ComputeDisksAddResourcePoliciesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddResourcePolicies')
        return self._RunMethod(config, request, global_params=global_params)
    AddResourcePolicies.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.disks.addResourcePolicies', ordered_params=['project', 'zone', 'disk'], path_params=['disk', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/disks/{disk}/addResourcePolicies', request_field='disksAddResourcePoliciesRequest', request_type_name='ComputeDisksAddResourcePoliciesRequest', response_type_name='Operation', supports_download=False)

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of persistent disks. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeDisksAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DiskAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.disks.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/disks', request_field='', request_type_name='ComputeDisksAggregatedListRequest', response_type_name='DiskAggregatedList', supports_download=False)

    def BulkInsert(self, request, global_params=None):
        """Bulk create a set of disks.

      Args:
        request: (ComputeDisksBulkInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('BulkInsert')
        return self._RunMethod(config, request, global_params=global_params)
    BulkInsert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.disks.bulkInsert', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/disks/bulkInsert', request_field='bulkInsertDiskResource', request_type_name='ComputeDisksBulkInsertRequest', response_type_name='Operation', supports_download=False)

    def CreateSnapshot(self, request, global_params=None):
        """Creates a snapshot of a specified persistent disk. For regular snapshot creation, consider using snapshots.insert instead, as that method supports more features, such as creating snapshots in a project different from the source disk project.

      Args:
        request: (ComputeDisksCreateSnapshotRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('CreateSnapshot')
        return self._RunMethod(config, request, global_params=global_params)
    CreateSnapshot.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.disks.createSnapshot', ordered_params=['project', 'zone', 'disk'], path_params=['disk', 'project', 'zone'], query_params=['guestFlush', 'requestId'], relative_path='projects/{project}/zones/{zone}/disks/{disk}/createSnapshot', request_field='snapshot', request_type_name='ComputeDisksCreateSnapshotRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified persistent disk. Deleting a disk removes its data permanently and is irreversible. However, deleting a disk does not delete any snapshots previously made from the disk. You must separately delete snapshots.

      Args:
        request: (ComputeDisksDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.disks.delete', ordered_params=['project', 'zone', 'disk'], path_params=['disk', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/disks/{disk}', request_field='', request_type_name='ComputeDisksDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified persistent disk.

      Args:
        request: (ComputeDisksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Disk) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.disks.get', ordered_params=['project', 'zone', 'disk'], path_params=['disk', 'project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/disks/{disk}', request_field='', request_type_name='ComputeDisksGetRequest', response_type_name='Disk', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeDisksGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.disks.getIamPolicy', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/zones/{zone}/disks/{resource}/getIamPolicy', request_field='', request_type_name='ComputeDisksGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a persistent disk in the specified project using the data in the request. You can create a disk from a source (sourceImage, sourceSnapshot, or sourceDisk) or create an empty 500 GB data disk by omitting all properties. You can also create a disk that is larger than the default size by specifying the sizeGb property.

      Args:
        request: (ComputeDisksInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.disks.insert', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['requestId', 'sourceImage'], relative_path='projects/{project}/zones/{zone}/disks', request_field='disk', request_type_name='ComputeDisksInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of persistent disks contained within the specified zone.

      Args:
        request: (ComputeDisksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DiskList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.disks.list', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/disks', request_field='', request_type_name='ComputeDisksListRequest', response_type_name='DiskList', supports_download=False)

    def RemoveResourcePolicies(self, request, global_params=None):
        """Removes resource policies from a disk.

      Args:
        request: (ComputeDisksRemoveResourcePoliciesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RemoveResourcePolicies')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveResourcePolicies.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.disks.removeResourcePolicies', ordered_params=['project', 'zone', 'disk'], path_params=['disk', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/disks/{disk}/removeResourcePolicies', request_field='disksRemoveResourcePoliciesRequest', request_type_name='ComputeDisksRemoveResourcePoliciesRequest', response_type_name='Operation', supports_download=False)

    def Resize(self, request, global_params=None):
        """Resizes the specified persistent disk. You can only increase the size of the disk.

      Args:
        request: (ComputeDisksResizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Resize')
        return self._RunMethod(config, request, global_params=global_params)
    Resize.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.disks.resize', ordered_params=['project', 'zone', 'disk'], path_params=['disk', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/disks/{disk}/resize', request_field='disksResizeRequest', request_type_name='ComputeDisksResizeRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeDisksSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.disks.setIamPolicy', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/disks/{resource}/setIamPolicy', request_field='zoneSetPolicyRequest', request_type_name='ComputeDisksSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def SetLabels(self, request, global_params=None):
        """Sets the labels on a disk. To learn more about labels, read the Labeling Resources documentation.

      Args:
        request: (ComputeDisksSetLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLabels')
        return self._RunMethod(config, request, global_params=global_params)
    SetLabels.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.disks.setLabels', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/disks/{resource}/setLabels', request_field='zoneSetLabelsRequest', request_type_name='ComputeDisksSetLabelsRequest', response_type_name='Operation', supports_download=False)

    def StartAsyncReplication(self, request, global_params=None):
        """Starts asynchronous replication. Must be invoked on the primary disk.

      Args:
        request: (ComputeDisksStartAsyncReplicationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('StartAsyncReplication')
        return self._RunMethod(config, request, global_params=global_params)
    StartAsyncReplication.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.disks.startAsyncReplication', ordered_params=['project', 'zone', 'disk'], path_params=['disk', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/disks/{disk}/startAsyncReplication', request_field='disksStartAsyncReplicationRequest', request_type_name='ComputeDisksStartAsyncReplicationRequest', response_type_name='Operation', supports_download=False)

    def StopAsyncReplication(self, request, global_params=None):
        """Stops asynchronous replication. Can be invoked either on the primary or on the secondary disk.

      Args:
        request: (ComputeDisksStopAsyncReplicationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('StopAsyncReplication')
        return self._RunMethod(config, request, global_params=global_params)
    StopAsyncReplication.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.disks.stopAsyncReplication', ordered_params=['project', 'zone', 'disk'], path_params=['disk', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/disks/{disk}/stopAsyncReplication', request_field='', request_type_name='ComputeDisksStopAsyncReplicationRequest', response_type_name='Operation', supports_download=False)

    def StopGroupAsyncReplication(self, request, global_params=None):
        """Stops asynchronous replication for a consistency group of disks. Can be invoked either in the primary or secondary scope.

      Args:
        request: (ComputeDisksStopGroupAsyncReplicationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('StopGroupAsyncReplication')
        return self._RunMethod(config, request, global_params=global_params)
    StopGroupAsyncReplication.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.disks.stopGroupAsyncReplication', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/disks/stopGroupAsyncReplication', request_field='disksStopGroupAsyncReplicationResource', request_type_name='ComputeDisksStopGroupAsyncReplicationRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeDisksTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.disks.testIamPermissions', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/disks/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeDisksTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the specified disk with the data included in the request. The update is performed only on selected fields included as part of update-mask. Only the following fields can be modified: user_license.

      Args:
        request: (ComputeDisksUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.disks.update', ordered_params=['project', 'zone', 'disk'], path_params=['disk', 'project', 'zone'], query_params=['paths', 'requestId', 'updateMask'], relative_path='projects/{project}/zones/{zone}/disks/{disk}', request_field='diskResource', request_type_name='ComputeDisksUpdateRequest', response_type_name='Operation', supports_download=False)