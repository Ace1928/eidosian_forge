from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class SslPoliciesService(base_api.BaseApiService):
    """Service class for the sslPolicies resource."""
    _NAME = 'sslPolicies'

    def __init__(self, client):
        super(ComputeBeta.SslPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves the list of all SslPolicy resources, regional and global, available to the specified project. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeSslPoliciesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SslPoliciesAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.sslPolicies.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/sslPolicies', request_field='', request_type_name='ComputeSslPoliciesAggregatedListRequest', response_type_name='SslPoliciesAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified SSL policy. The SSL policy resource can be deleted only if it is not in use by any TargetHttpsProxy or TargetSslProxy resources.

      Args:
        request: (ComputeSslPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.sslPolicies.delete', ordered_params=['project', 'sslPolicy'], path_params=['project', 'sslPolicy'], query_params=['requestId'], relative_path='projects/{project}/global/sslPolicies/{sslPolicy}', request_field='', request_type_name='ComputeSslPoliciesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Lists all of the ordered rules present in a single specified policy.

      Args:
        request: (ComputeSslPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SslPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.sslPolicies.get', ordered_params=['project', 'sslPolicy'], path_params=['project', 'sslPolicy'], query_params=[], relative_path='projects/{project}/global/sslPolicies/{sslPolicy}', request_field='', request_type_name='ComputeSslPoliciesGetRequest', response_type_name='SslPolicy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Returns the specified SSL policy resource.

      Args:
        request: (ComputeSslPoliciesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.sslPolicies.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/sslPolicies', request_field='sslPolicy', request_type_name='ComputeSslPoliciesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all the SSL policies that have been configured for the specified project.

      Args:
        request: (ComputeSslPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SslPoliciesList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.sslPolicies.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/sslPolicies', request_field='', request_type_name='ComputeSslPoliciesListRequest', response_type_name='SslPoliciesList', supports_download=False)

    def ListAvailableFeatures(self, request, global_params=None):
        """Lists all features that can be specified in the SSL policy when using custom profile.

      Args:
        request: (ComputeSslPoliciesListAvailableFeaturesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SslPoliciesListAvailableFeaturesResponse) The response message.
      """
        config = self.GetMethodConfig('ListAvailableFeatures')
        return self._RunMethod(config, request, global_params=global_params)
    ListAvailableFeatures.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.sslPolicies.listAvailableFeatures', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/sslPolicies/listAvailableFeatures', request_field='', request_type_name='ComputeSslPoliciesListAvailableFeaturesRequest', response_type_name='SslPoliciesListAvailableFeaturesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified SSL policy with the data included in the request.

      Args:
        request: (ComputeSslPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.sslPolicies.patch', ordered_params=['project', 'sslPolicy'], path_params=['project', 'sslPolicy'], query_params=['requestId'], relative_path='projects/{project}/global/sslPolicies/{sslPolicy}', request_field='sslPolicyResource', request_type_name='ComputeSslPoliciesPatchRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeSslPoliciesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.sslPolicies.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/sslPolicies/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeSslPoliciesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)