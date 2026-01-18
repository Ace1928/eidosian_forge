from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigquery.v2 import bigquery_v2_messages as messages
class RowAccessPoliciesService(base_api.BaseApiService):
    """Service class for the rowAccessPolicies resource."""
    _NAME = 'rowAccessPolicies'

    def __init__(self, client):
        super(BigqueryV2.RowAccessPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (BigqueryRowAccessPoliciesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/tables/{tablesId}/rowAccessPolicies/{rowAccessPoliciesId}:getIamPolicy', http_method='POST', method_id='bigquery.rowAccessPolicies.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='BigqueryRowAccessPoliciesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all row access policies on the specified table.

      Args:
        request: (BigqueryRowAccessPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRowAccessPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/tables/{tablesId}/rowAccessPolicies', http_method='GET', method_id='bigquery.rowAccessPolicies.list', ordered_params=['projectId', 'datasetId', 'tableId'], path_params=['datasetId', 'projectId', 'tableId'], query_params=['pageSize', 'pageToken'], relative_path='projects/{+projectId}/datasets/{+datasetId}/tables/{+tableId}/rowAccessPolicies', request_field='', request_type_name='BigqueryRowAccessPoliciesListRequest', response_type_name='ListRowAccessPoliciesResponse', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (BigqueryRowAccessPoliciesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/tables/{tablesId}/rowAccessPolicies/{rowAccessPoliciesId}:testIamPermissions', http_method='POST', method_id='bigquery.rowAccessPolicies.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='BigqueryRowAccessPoliciesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)