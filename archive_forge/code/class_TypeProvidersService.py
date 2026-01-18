from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.deploymentmanager.alpha import deploymentmanager_alpha_messages as messages
class TypeProvidersService(base_api.BaseApiService):
    """Service class for the typeProviders resource."""
    _NAME = 'typeProviders'

    def __init__(self, client):
        super(DeploymentmanagerAlpha.TypeProvidersService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a type provider.

      Args:
        request: (DeploymentmanagerTypeProvidersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='deploymentmanager.typeProviders.delete', ordered_params=['project', 'typeProvider'], path_params=['project', 'typeProvider'], query_params=[], relative_path='deploymentmanager/alpha/projects/{project}/global/typeProviders/{typeProvider}', request_field='', request_type_name='DeploymentmanagerTypeProvidersDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a specific type provider.

      Args:
        request: (DeploymentmanagerTypeProvidersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TypeProvider) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='deploymentmanager.typeProviders.get', ordered_params=['project', 'typeProvider'], path_params=['project', 'typeProvider'], query_params=[], relative_path='deploymentmanager/alpha/projects/{project}/global/typeProviders/{typeProvider}', request_field='', request_type_name='DeploymentmanagerTypeProvidersGetRequest', response_type_name='TypeProvider', supports_download=False)

    def GetType(self, request, global_params=None):
        """Gets a type info for a type provided by a TypeProvider.

      Args:
        request: (DeploymentmanagerTypeProvidersGetTypeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TypeInfo) The response message.
      """
        config = self.GetMethodConfig('GetType')
        return self._RunMethod(config, request, global_params=global_params)
    GetType.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='deploymentmanager.typeProviders.getType', ordered_params=['project', 'typeProvider', 'type'], path_params=['project', 'type', 'typeProvider'], query_params=[], relative_path='deploymentmanager/alpha/projects/{project}/global/typeProviders/{typeProvider}/types/{type}', request_field='', request_type_name='DeploymentmanagerTypeProvidersGetTypeRequest', response_type_name='TypeInfo', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a type provider.

      Args:
        request: (DeploymentmanagerTypeProvidersInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='deploymentmanager.typeProviders.insert', ordered_params=['project'], path_params=['project'], query_params=[], relative_path='deploymentmanager/alpha/projects/{project}/global/typeProviders', request_field='typeProvider', request_type_name='DeploymentmanagerTypeProvidersInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all resource type providers for Deployment Manager.

      Args:
        request: (DeploymentmanagerTypeProvidersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TypeProvidersListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='deploymentmanager.typeProviders.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken'], relative_path='deploymentmanager/alpha/projects/{project}/global/typeProviders', request_field='', request_type_name='DeploymentmanagerTypeProvidersListRequest', response_type_name='TypeProvidersListResponse', supports_download=False)

    def ListTypes(self, request, global_params=None):
        """Lists all the type info for a TypeProvider.

      Args:
        request: (DeploymentmanagerTypeProvidersListTypesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TypeProvidersListTypesResponse) The response message.
      """
        config = self.GetMethodConfig('ListTypes')
        return self._RunMethod(config, request, global_params=global_params)
    ListTypes.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='deploymentmanager.typeProviders.listTypes', ordered_params=['project', 'typeProvider'], path_params=['project', 'typeProvider'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken'], relative_path='deploymentmanager/alpha/projects/{project}/global/typeProviders/{typeProvider}/types', request_field='', request_type_name='DeploymentmanagerTypeProvidersListTypesRequest', response_type_name='TypeProvidersListTypesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches a type provider.

      Args:
        request: (DeploymentmanagerTypeProvidersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='deploymentmanager.typeProviders.patch', ordered_params=['project', 'typeProvider'], path_params=['project', 'typeProvider'], query_params=[], relative_path='deploymentmanager/alpha/projects/{project}/global/typeProviders/{typeProvider}', request_field='typeProviderResource', request_type_name='DeploymentmanagerTypeProvidersPatchRequest', response_type_name='Operation', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a type provider.

      Args:
        request: (DeploymentmanagerTypeProvidersUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='deploymentmanager.typeProviders.update', ordered_params=['project', 'typeProvider'], path_params=['project', 'typeProvider'], query_params=[], relative_path='deploymentmanager/alpha/projects/{project}/global/typeProviders/{typeProvider}', request_field='typeProviderResource', request_type_name='DeploymentmanagerTypeProvidersUpdateRequest', response_type_name='Operation', supports_download=False)