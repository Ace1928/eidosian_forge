from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
class FoldersLocationsBigQueryExportsService(base_api.BaseApiService):
    """Service class for the folders_locations_bigQueryExports resource."""
    _NAME = 'folders_locations_bigQueryExports'

    def __init__(self, client):
        super(SecuritycenterV2.FoldersLocationsBigQueryExportsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a BigQuery export.

      Args:
        request: (SecuritycenterFoldersLocationsBigQueryExportsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV2BigQueryExport) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/folders/{foldersId}/locations/{locationsId}/bigQueryExports', http_method='POST', method_id='securitycenter.folders.locations.bigQueryExports.create', ordered_params=['parent'], path_params=['parent'], query_params=['bigQueryExportId'], relative_path='v2/{+parent}/bigQueryExports', request_field='googleCloudSecuritycenterV2BigQueryExport', request_type_name='SecuritycenterFoldersLocationsBigQueryExportsCreateRequest', response_type_name='GoogleCloudSecuritycenterV2BigQueryExport', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an existing BigQuery export.

      Args:
        request: (SecuritycenterFoldersLocationsBigQueryExportsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/folders/{foldersId}/locations/{locationsId}/bigQueryExports/{bigQueryExportsId}', http_method='DELETE', method_id='securitycenter.folders.locations.bigQueryExports.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='SecuritycenterFoldersLocationsBigQueryExportsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a BigQuery export.

      Args:
        request: (SecuritycenterFoldersLocationsBigQueryExportsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV2BigQueryExport) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/folders/{foldersId}/locations/{locationsId}/bigQueryExports/{bigQueryExportsId}', http_method='GET', method_id='securitycenter.folders.locations.bigQueryExports.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='SecuritycenterFoldersLocationsBigQueryExportsGetRequest', response_type_name='GoogleCloudSecuritycenterV2BigQueryExport', supports_download=False)

    def List(self, request, global_params=None):
        """Lists BigQuery exports. Note that when requesting BigQuery exports at a given level all exports under that level are also returned e.g. if requesting BigQuery exports under a folder, then all BigQuery exports immediately under the folder plus the ones created under the projects within the folder are returned.

      Args:
        request: (SecuritycenterFoldersLocationsBigQueryExportsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBigQueryExportsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/folders/{foldersId}/locations/{locationsId}/bigQueryExports', http_method='GET', method_id='securitycenter.folders.locations.bigQueryExports.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/bigQueryExports', request_field='', request_type_name='SecuritycenterFoldersLocationsBigQueryExportsListRequest', response_type_name='ListBigQueryExportsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a BigQuery export.

      Args:
        request: (SecuritycenterFoldersLocationsBigQueryExportsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV2BigQueryExport) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/folders/{foldersId}/locations/{locationsId}/bigQueryExports/{bigQueryExportsId}', http_method='PATCH', method_id='securitycenter.folders.locations.bigQueryExports.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudSecuritycenterV2BigQueryExport', request_type_name='SecuritycenterFoldersLocationsBigQueryExportsPatchRequest', response_type_name='GoogleCloudSecuritycenterV2BigQueryExport', supports_download=False)