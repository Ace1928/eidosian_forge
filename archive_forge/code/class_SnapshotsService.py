from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class SnapshotsService(base_api.BaseApiService):
    """Service class for the snapshots resource."""
    _NAME = 'snapshots'

    def __init__(self, client):
        super(ComputeBeta.SnapshotsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified Snapshot resource. Keep in mind that deleting a single snapshot might not necessarily delete all the data on that snapshot. If any data on the snapshot that is marked for deletion is needed for subsequent snapshots, the data will be moved to the next corresponding snapshot. For more information, see Deleting snapshots.

      Args:
        request: (ComputeSnapshotsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.snapshots.delete', ordered_params=['project', 'snapshot'], path_params=['project', 'snapshot'], query_params=['requestId'], relative_path='projects/{project}/global/snapshots/{snapshot}', request_field='', request_type_name='ComputeSnapshotsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified Snapshot resource.

      Args:
        request: (ComputeSnapshotsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Snapshot) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.snapshots.get', ordered_params=['project', 'snapshot'], path_params=['project', 'snapshot'], query_params=[], relative_path='projects/{project}/global/snapshots/{snapshot}', request_field='', request_type_name='ComputeSnapshotsGetRequest', response_type_name='Snapshot', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeSnapshotsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.snapshots.getIamPolicy', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/global/snapshots/{resource}/getIamPolicy', request_field='', request_type_name='ComputeSnapshotsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a snapshot in the specified project using the data included in the request. For regular snapshot creation, consider using this method instead of disks.createSnapshot, as this method supports more features, such as creating snapshots in a project different from the source disk project.

      Args:
        request: (ComputeSnapshotsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.snapshots.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/snapshots', request_field='snapshot', request_type_name='ComputeSnapshotsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of Snapshot resources contained within the specified project.

      Args:
        request: (ComputeSnapshotsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SnapshotList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.snapshots.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/snapshots', request_field='', request_type_name='ComputeSnapshotsListRequest', response_type_name='SnapshotList', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeSnapshotsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.snapshots.setIamPolicy', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/snapshots/{resource}/setIamPolicy', request_field='globalSetPolicyRequest', request_type_name='ComputeSnapshotsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def SetLabels(self, request, global_params=None):
        """Sets the labels on a snapshot. To learn more about labels, read the Labeling Resources documentation.

      Args:
        request: (ComputeSnapshotsSetLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLabels')
        return self._RunMethod(config, request, global_params=global_params)
    SetLabels.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.snapshots.setLabels', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/snapshots/{resource}/setLabels', request_field='globalSetLabelsRequest', request_type_name='ComputeSnapshotsSetLabelsRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeSnapshotsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.snapshots.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/snapshots/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeSnapshotsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)