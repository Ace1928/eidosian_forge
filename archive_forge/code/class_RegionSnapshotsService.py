from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.alpha import compute_alpha_messages as messages
class RegionSnapshotsService(base_api.BaseApiService):
    """Service class for the regionSnapshots resource."""
    _NAME = 'regionSnapshots'

    def __init__(self, client):
        super(ComputeAlpha.RegionSnapshotsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified Snapshot resource. Keep in mind that deleting a single snapshot might not necessarily delete all the data on that snapshot. If any data on the snapshot that is marked for deletion is needed for subsequent snapshots, the data will be moved to the next corresponding snapshot. For more information, see Deleting snapshots.

      Args:
        request: (ComputeRegionSnapshotsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.regionSnapshots.delete', ordered_params=['project', 'region', 'snapshot'], path_params=['project', 'region', 'snapshot'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/snapshots/{snapshot}', request_field='', request_type_name='ComputeRegionSnapshotsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified Snapshot resource.

      Args:
        request: (ComputeRegionSnapshotsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Snapshot) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionSnapshots.get', ordered_params=['project', 'region', 'snapshot'], path_params=['project', 'region', 'snapshot'], query_params=[], relative_path='projects/{project}/regions/{region}/snapshots/{snapshot}', request_field='', request_type_name='ComputeRegionSnapshotsGetRequest', response_type_name='Snapshot', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeRegionSnapshotsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionSnapshots.getIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/regions/{region}/snapshots/{resource}/getIamPolicy', request_field='', request_type_name='ComputeRegionSnapshotsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a snapshot in the specified region using the data included in the request.

      Args:
        request: (ComputeRegionSnapshotsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionSnapshots.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/snapshots', request_field='snapshot', request_type_name='ComputeRegionSnapshotsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of Snapshot resources contained within the specified region.

      Args:
        request: (ComputeRegionSnapshotsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SnapshotList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionSnapshots.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/snapshots', request_field='', request_type_name='ComputeRegionSnapshotsListRequest', response_type_name='SnapshotList', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeRegionSnapshotsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionSnapshots.setIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/snapshots/{resource}/setIamPolicy', request_field='regionSetPolicyRequest', request_type_name='ComputeRegionSnapshotsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def SetLabels(self, request, global_params=None):
        """Sets the labels on a regional snapshot. To learn more about labels, read the Labeling Resources documentation.

      Args:
        request: (ComputeRegionSnapshotsSetLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLabels')
        return self._RunMethod(config, request, global_params=global_params)
    SetLabels.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionSnapshots.setLabels', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/snapshots/{resource}/setLabels', request_field='regionSetLabelsRequest', request_type_name='ComputeRegionSnapshotsSetLabelsRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeRegionSnapshotsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionSnapshots.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/snapshots/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRegionSnapshotsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)