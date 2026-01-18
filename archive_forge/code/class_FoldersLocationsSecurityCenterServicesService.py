from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycentermanagement.v1 import securitycentermanagement_v1_messages as messages
class FoldersLocationsSecurityCenterServicesService(base_api.BaseApiService):
    """Service class for the folders_locations_securityCenterServices resource."""
    _NAME = 'folders_locations_securityCenterServices'

    def __init__(self, client):
        super(SecuritycentermanagementV1.FoldersLocationsSecurityCenterServicesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets service settings for the specified Security Center service.

      Args:
        request: (SecuritycentermanagementFoldersLocationsSecurityCenterServicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityCenterService) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/locations/{locationsId}/securityCenterServices/{securityCenterServicesId}', http_method='GET', method_id='securitycentermanagement.folders.locations.securityCenterServices.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SecuritycentermanagementFoldersLocationsSecurityCenterServicesGetRequest', response_type_name='SecurityCenterService', supports_download=False)

    def List(self, request, global_params=None):
        """Returns a list of all Security Center services for the given parent.

      Args:
        request: (SecuritycentermanagementFoldersLocationsSecurityCenterServicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSecurityCenterServicesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/locations/{locationsId}/securityCenterServices', http_method='GET', method_id='securitycentermanagement.folders.locations.securityCenterServices.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/securityCenterServices', request_field='', request_type_name='SecuritycentermanagementFoldersLocationsSecurityCenterServicesListRequest', response_type_name='ListSecurityCenterServicesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a Security Center service using the given update mask.

      Args:
        request: (SecuritycentermanagementFoldersLocationsSecurityCenterServicesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityCenterService) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/locations/{locationsId}/securityCenterServices/{securityCenterServicesId}', http_method='PATCH', method_id='securitycentermanagement.folders.locations.securityCenterServices.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='securityCenterService', request_type_name='SecuritycentermanagementFoldersLocationsSecurityCenterServicesPatchRequest', response_type_name='SecurityCenterService', supports_download=False)