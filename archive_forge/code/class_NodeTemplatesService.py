from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class NodeTemplatesService(base_api.BaseApiService):
    """Service class for the nodeTemplates resource."""
    _NAME = 'nodeTemplates'

    def __init__(self, client):
        super(ComputeBeta.NodeTemplatesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of node templates. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeNodeTemplatesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NodeTemplateAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.nodeTemplates.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/nodeTemplates', request_field='', request_type_name='ComputeNodeTemplatesAggregatedListRequest', response_type_name='NodeTemplateAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified NodeTemplate resource.

      Args:
        request: (ComputeNodeTemplatesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.nodeTemplates.delete', ordered_params=['project', 'region', 'nodeTemplate'], path_params=['nodeTemplate', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/nodeTemplates/{nodeTemplate}', request_field='', request_type_name='ComputeNodeTemplatesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified node template.

      Args:
        request: (ComputeNodeTemplatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NodeTemplate) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.nodeTemplates.get', ordered_params=['project', 'region', 'nodeTemplate'], path_params=['nodeTemplate', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/nodeTemplates/{nodeTemplate}', request_field='', request_type_name='ComputeNodeTemplatesGetRequest', response_type_name='NodeTemplate', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeNodeTemplatesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.nodeTemplates.getIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/regions/{region}/nodeTemplates/{resource}/getIamPolicy', request_field='', request_type_name='ComputeNodeTemplatesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a NodeTemplate resource in the specified project using the data included in the request.

      Args:
        request: (ComputeNodeTemplatesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.nodeTemplates.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/nodeTemplates', request_field='nodeTemplate', request_type_name='ComputeNodeTemplatesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of node templates available to the specified project.

      Args:
        request: (ComputeNodeTemplatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NodeTemplateList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.nodeTemplates.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/nodeTemplates', request_field='', request_type_name='ComputeNodeTemplatesListRequest', response_type_name='NodeTemplateList', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeNodeTemplatesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.nodeTemplates.setIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/nodeTemplates/{resource}/setIamPolicy', request_field='regionSetPolicyRequest', request_type_name='ComputeNodeTemplatesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeNodeTemplatesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.nodeTemplates.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/nodeTemplates/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeNodeTemplatesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)