from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class ExternalVpnGatewaysService(base_api.BaseApiService):
    """Service class for the externalVpnGateways resource."""
    _NAME = 'externalVpnGateways'

    def __init__(self, client):
        super(ComputeBeta.ExternalVpnGatewaysService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified externalVpnGateway.

      Args:
        request: (ComputeExternalVpnGatewaysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.externalVpnGateways.delete', ordered_params=['project', 'externalVpnGateway'], path_params=['externalVpnGateway', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/externalVpnGateways/{externalVpnGateway}', request_field='', request_type_name='ComputeExternalVpnGatewaysDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified externalVpnGateway. Get a list of available externalVpnGateways by making a list() request.

      Args:
        request: (ComputeExternalVpnGatewaysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExternalVpnGateway) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.externalVpnGateways.get', ordered_params=['project', 'externalVpnGateway'], path_params=['externalVpnGateway', 'project'], query_params=[], relative_path='projects/{project}/global/externalVpnGateways/{externalVpnGateway}', request_field='', request_type_name='ComputeExternalVpnGatewaysGetRequest', response_type_name='ExternalVpnGateway', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a ExternalVpnGateway in the specified project using the data included in the request.

      Args:
        request: (ComputeExternalVpnGatewaysInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.externalVpnGateways.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/externalVpnGateways', request_field='externalVpnGateway', request_type_name='ComputeExternalVpnGatewaysInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of ExternalVpnGateway available to the specified project.

      Args:
        request: (ComputeExternalVpnGatewaysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExternalVpnGatewayList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.externalVpnGateways.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/externalVpnGateways', request_field='', request_type_name='ComputeExternalVpnGatewaysListRequest', response_type_name='ExternalVpnGatewayList', supports_download=False)

    def SetLabels(self, request, global_params=None):
        """Sets the labels on an ExternalVpnGateway. To learn more about labels, read the Labeling Resources documentation.

      Args:
        request: (ComputeExternalVpnGatewaysSetLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLabels')
        return self._RunMethod(config, request, global_params=global_params)
    SetLabels.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.externalVpnGateways.setLabels', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/externalVpnGateways/{resource}/setLabels', request_field='globalSetLabelsRequest', request_type_name='ComputeExternalVpnGatewaysSetLabelsRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeExternalVpnGatewaysTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.externalVpnGateways.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/externalVpnGateways/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeExternalVpnGatewaysTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)