from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class ResourcesBuildingsService(base_api.BaseApiService):
    """Service class for the resources_buildings resource."""
    _NAME = u'resources_buildings'

    def __init__(self, client):
        super(AdminDirectoryV1.ResourcesBuildingsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a building.

      Args:
        request: (DirectoryResourcesBuildingsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryResourcesBuildingsDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'directory.resources.buildings.delete', ordered_params=[u'customer', u'buildingId'], path_params=[u'buildingId', u'customer'], query_params=[], relative_path=u'customer/{customer}/resources/buildings/{buildingId}', request_field='', request_type_name=u'DirectoryResourcesBuildingsDeleteRequest', response_type_name=u'DirectoryResourcesBuildingsDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a building.

      Args:
        request: (DirectoryResourcesBuildingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Building) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.resources.buildings.get', ordered_params=[u'customer', u'buildingId'], path_params=[u'buildingId', u'customer'], query_params=[], relative_path=u'customer/{customer}/resources/buildings/{buildingId}', request_field='', request_type_name=u'DirectoryResourcesBuildingsGetRequest', response_type_name=u'Building', supports_download=False)

    def Insert(self, request, global_params=None):
        """Inserts a building.

      Args:
        request: (DirectoryResourcesBuildingsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Building) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.resources.buildings.insert', ordered_params=[u'customer'], path_params=[u'customer'], query_params=[u'coordinatesSource'], relative_path=u'customer/{customer}/resources/buildings', request_field=u'building', request_type_name=u'DirectoryResourcesBuildingsInsertRequest', response_type_name=u'Building', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of buildings for an account.

      Args:
        request: (DirectoryResourcesBuildingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Buildings) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.resources.buildings.list', ordered_params=[u'customer'], path_params=[u'customer'], query_params=[u'maxResults', u'pageToken'], relative_path=u'customer/{customer}/resources/buildings', request_field='', request_type_name=u'DirectoryResourcesBuildingsListRequest', response_type_name=u'Buildings', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a building.

      This method supports patch semantics.

      Args:
        request: (DirectoryResourcesBuildingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Building) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'directory.resources.buildings.patch', ordered_params=[u'customer', u'buildingId'], path_params=[u'buildingId', u'customer'], query_params=[u'coordinatesSource'], relative_path=u'customer/{customer}/resources/buildings/{buildingId}', request_field=u'building', request_type_name=u'DirectoryResourcesBuildingsPatchRequest', response_type_name=u'Building', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a building.

      Args:
        request: (DirectoryResourcesBuildingsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Building) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'directory.resources.buildings.update', ordered_params=[u'customer', u'buildingId'], path_params=[u'buildingId', u'customer'], query_params=[u'coordinatesSource'], relative_path=u'customer/{customer}/resources/buildings/{buildingId}', request_field=u'building', request_type_name=u'DirectoryResourcesBuildingsUpdateRequest', response_type_name=u'Building', supports_download=False)