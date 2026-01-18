from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1alpha1 import osconfig_v1alpha1_messages as messages
class FoldersOsConfigsService(base_api.BaseApiService):
    """Service class for the folders_osConfigs resource."""
    _NAME = 'folders_osConfigs'

    def __init__(self, client):
        super(OsconfigV1alpha1.FoldersOsConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create an OsConfig.

      Args:
        request: (OsconfigFoldersOsConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OsConfig) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/folders/{foldersId}/osConfigs', http_method='POST', method_id='osconfig.folders.osConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/osConfigs', request_field='osConfig', request_type_name='OsconfigFoldersOsConfigsCreateRequest', response_type_name='OsConfig', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete an OsConfig.

      Args:
        request: (OsconfigFoldersOsConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/folders/{foldersId}/osConfigs/{osConfigsId}', http_method='DELETE', method_id='osconfig.folders.osConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='OsconfigFoldersOsConfigsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Get an OsConfig.

      Args:
        request: (OsconfigFoldersOsConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OsConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/folders/{foldersId}/osConfigs/{osConfigsId}', http_method='GET', method_id='osconfig.folders.osConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='OsconfigFoldersOsConfigsGetRequest', response_type_name='OsConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Get a page of OsConfigs.

      Args:
        request: (OsconfigFoldersOsConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOsConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/folders/{foldersId}/osConfigs', http_method='GET', method_id='osconfig.folders.osConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/osConfigs', request_field='', request_type_name='OsconfigFoldersOsConfigsListRequest', response_type_name='ListOsConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update an OsConfig.

      Args:
        request: (OsconfigFoldersOsConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OsConfig) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/folders/{foldersId}/osConfigs/{osConfigsId}', http_method='PATCH', method_id='osconfig.folders.osConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='osConfig', request_type_name='OsconfigFoldersOsConfigsPatchRequest', response_type_name='OsConfig', supports_download=False)