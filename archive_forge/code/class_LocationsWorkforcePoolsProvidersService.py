from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1 import iam_v1_messages as messages
class LocationsWorkforcePoolsProvidersService(base_api.BaseApiService):
    """Service class for the locations_workforcePools_providers resource."""
    _NAME = 'locations_workforcePools_providers'

    def __init__(self, client):
        super(IamV1.LocationsWorkforcePoolsProvidersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new WorkforcePoolProvider in a WorkforcePool. You cannot reuse the name of a deleted provider until 30 days after deletion.

      Args:
        request: (IamLocationsWorkforcePoolsProvidersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/providers', http_method='POST', method_id='iam.locations.workforcePools.providers.create', ordered_params=['parent'], path_params=['parent'], query_params=['workforcePoolProviderId'], relative_path='v1/{+parent}/providers', request_field='workforcePoolProvider', request_type_name='IamLocationsWorkforcePoolsProvidersCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a WorkforcePoolProvider. Deleting a provider does not revoke credentials that have already been\\ issued; they continue to grant access. You can undelete a provider for 30 days. After 30 days, deletion is permanent. You cannot update deleted providers. However, you can view and list them.

      Args:
        request: (IamLocationsWorkforcePoolsProvidersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/providers/{providersId}', http_method='DELETE', method_id='iam.locations.workforcePools.providers.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamLocationsWorkforcePoolsProvidersDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an individual WorkforcePoolProvider.

      Args:
        request: (IamLocationsWorkforcePoolsProvidersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkforcePoolProvider) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/providers/{providersId}', http_method='GET', method_id='iam.locations.workforcePools.providers.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamLocationsWorkforcePoolsProvidersGetRequest', response_type_name='WorkforcePoolProvider', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all non-deleted WorkforcePoolProviders in a WorkforcePool. If `show_deleted` is set to `true`, then deleted providers are also listed.

      Args:
        request: (IamLocationsWorkforcePoolsProvidersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkforcePoolProvidersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/providers', http_method='GET', method_id='iam.locations.workforcePools.providers.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted'], relative_path='v1/{+parent}/providers', request_field='', request_type_name='IamLocationsWorkforcePoolsProvidersListRequest', response_type_name='ListWorkforcePoolProvidersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing WorkforcePoolProvider.

      Args:
        request: (IamLocationsWorkforcePoolsProvidersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/providers/{providersId}', http_method='PATCH', method_id='iam.locations.workforcePools.providers.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='workforcePoolProvider', request_type_name='IamLocationsWorkforcePoolsProvidersPatchRequest', response_type_name='Operation', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes a WorkforcePoolProvider, as long as it was deleted fewer than 30 days ago.

      Args:
        request: (IamLocationsWorkforcePoolsProvidersUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/providers/{providersId}:undelete', http_method='POST', method_id='iam.locations.workforcePools.providers.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:undelete', request_field='undeleteWorkforcePoolProviderRequest', request_type_name='IamLocationsWorkforcePoolsProvidersUndeleteRequest', response_type_name='Operation', supports_download=False)