from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionCommitmentsService(base_api.BaseApiService):
    """Service class for the regionCommitments resource."""
    _NAME = 'regionCommitments'

    def __init__(self, client):
        super(ComputeBeta.RegionCommitmentsService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of commitments by region. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeRegionCommitmentsAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CommitmentAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionCommitments.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/commitments', request_field='', request_type_name='ComputeRegionCommitmentsAggregatedListRequest', response_type_name='CommitmentAggregatedList', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified commitment resource.

      Args:
        request: (ComputeRegionCommitmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Commitment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionCommitments.get', ordered_params=['project', 'region', 'commitment'], path_params=['commitment', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/commitments/{commitment}', request_field='', request_type_name='ComputeRegionCommitmentsGetRequest', response_type_name='Commitment', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a commitment in the specified project using the data included in the request.

      Args:
        request: (ComputeRegionCommitmentsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionCommitments.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/commitments', request_field='commitment', request_type_name='ComputeRegionCommitmentsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of commitments contained within the specified region.

      Args:
        request: (ComputeRegionCommitmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CommitmentList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionCommitments.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/commitments', request_field='', request_type_name='ComputeRegionCommitmentsListRequest', response_type_name='CommitmentList', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeRegionCommitmentsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionCommitments.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/commitments/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRegionCommitmentsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the specified commitment with the data included in the request. Update is performed only on selected fields included as part of update-mask. Only the following fields can be modified: auto_renew.

      Args:
        request: (ComputeRegionCommitmentsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.regionCommitments.update', ordered_params=['project', 'region', 'commitment'], path_params=['commitment', 'project', 'region'], query_params=['paths', 'requestId', 'updateMask'], relative_path='projects/{project}/regions/{region}/commitments/{commitment}', request_field='commitmentResource', request_type_name='ComputeRegionCommitmentsUpdateRequest', response_type_name='Operation', supports_download=False)

    def UpdateReservations(self, request, global_params=None):
        """Transfers GPUs or local SSDs between reservations within commitments.

      Args:
        request: (ComputeRegionCommitmentsUpdateReservationsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('UpdateReservations')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateReservations.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionCommitments.updateReservations', ordered_params=['project', 'region', 'commitment'], path_params=['commitment', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/commitments/{commitment}/updateReservations', request_field='regionCommitmentsUpdateReservationsRequest', request_type_name='ComputeRegionCommitmentsUpdateReservationsRequest', response_type_name='Operation', supports_download=False)