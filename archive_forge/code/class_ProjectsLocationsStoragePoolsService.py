from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.netapp.v1 import netapp_v1_messages as messages
class ProjectsLocationsStoragePoolsService(base_api.BaseApiService):
    """Service class for the projects_locations_storagePools resource."""
    _NAME = 'projects_locations_storagePools'

    def __init__(self, client):
        super(NetappV1.ProjectsLocationsStoragePoolsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new storage pool.

      Args:
        request: (NetappProjectsLocationsStoragePoolsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/storagePools', http_method='POST', method_id='netapp.projects.locations.storagePools.create', ordered_params=['parent'], path_params=['parent'], query_params=['storagePoolId'], relative_path='v1/{+parent}/storagePools', request_field='storagePool', request_type_name='NetappProjectsLocationsStoragePoolsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Warning! This operation will permanently delete the storage pool.

      Args:
        request: (NetappProjectsLocationsStoragePoolsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/storagePools/{storagePoolsId}', http_method='DELETE', method_id='netapp.projects.locations.storagePools.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetappProjectsLocationsStoragePoolsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the description of the specified storage pool by poolId.

      Args:
        request: (NetappProjectsLocationsStoragePoolsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StoragePool) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/storagePools/{storagePoolsId}', http_method='GET', method_id='netapp.projects.locations.storagePools.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetappProjectsLocationsStoragePoolsGetRequest', response_type_name='StoragePool', supports_download=False)

    def List(self, request, global_params=None):
        """Returns descriptions of all storage pools owned by the caller.

      Args:
        request: (NetappProjectsLocationsStoragePoolsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListStoragePoolsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/storagePools', http_method='GET', method_id='netapp.projects.locations.storagePools.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/storagePools', request_field='', request_type_name='NetappProjectsLocationsStoragePoolsListRequest', response_type_name='ListStoragePoolsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the storage pool properties with the full spec.

      Args:
        request: (NetappProjectsLocationsStoragePoolsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/storagePools/{storagePoolsId}', http_method='PATCH', method_id='netapp.projects.locations.storagePools.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='storagePool', request_type_name='NetappProjectsLocationsStoragePoolsPatchRequest', response_type_name='Operation', supports_download=False)