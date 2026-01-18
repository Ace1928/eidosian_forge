from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v3beta import iam_v3beta_messages as messages
class FoldersLocationsService(base_api.BaseApiService):
    """Service class for the folders_locations resource."""
    _NAME = 'folders_locations'

    def __init__(self, client):
        super(IamV3beta.FoldersLocationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets information about a location.

      Args:
        request: (IamFoldersLocationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudLocationLocation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/folders/{foldersId}/locations/{locationsId}', http_method='GET', method_id='iam.folders.locations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3beta/{+name}', request_field='', request_type_name='IamFoldersLocationsGetRequest', response_type_name='GoogleCloudLocationLocation', supports_download=False)

    def ListLocations(self, request, global_params=None):
        """Lists information about the supported locations for this service.

      Args:
        request: (IamFoldersLocationsListLocationsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudLocationListLocationsResponse) The response message.
      """
        config = self.GetMethodConfig('ListLocations')
        return self._RunMethod(config, request, global_params=global_params)
    ListLocations.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/folders/{foldersId}/locations/{locationsId}', http_method='GET', method_id='iam.folders.locations.listLocations', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v3beta/{+name}', request_field='', request_type_name='IamFoldersLocationsListLocationsRequest', response_type_name='GoogleCloudLocationListLocationsResponse', supports_download=False)