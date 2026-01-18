from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmmigration.v1 import vmmigration_v1_messages as messages
class ProjectsLocationsSourcesService(base_api.BaseApiService):
    """Service class for the projects_locations_sources resource."""
    _NAME = 'projects_locations_sources'

    def __init__(self, client):
        super(VmmigrationV1.ProjectsLocationsSourcesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Source in a given project and location.

      Args:
        request: (VmmigrationProjectsLocationsSourcesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources', http_method='POST', method_id='vmmigration.projects.locations.sources.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'sourceId'], relative_path='v1/{+parent}/sources', request_field='source', request_type_name='VmmigrationProjectsLocationsSourcesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Source.

      Args:
        request: (VmmigrationProjectsLocationsSourcesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}', http_method='DELETE', method_id='vmmigration.projects.locations.sources.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesDeleteRequest', response_type_name='Operation', supports_download=False)

    def FetchInventory(self, request, global_params=None):
        """List remote source's inventory of VMs. The remote source is the onprem vCenter (remote in the sense it's not in Compute Engine). The inventory describes the list of existing VMs in that source. Note that this operation lists the VMs on the remote source, as opposed to listing the MigratingVms resources in the vmmigration service.

      Args:
        request: (VmmigrationProjectsLocationsSourcesFetchInventoryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FetchInventoryResponse) The response message.
      """
        config = self.GetMethodConfig('FetchInventory')
        return self._RunMethod(config, request, global_params=global_params)
    FetchInventory.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}:fetchInventory', http_method='GET', method_id='vmmigration.projects.locations.sources.fetchInventory', ordered_params=['source'], path_params=['source'], query_params=['forceRefresh', 'pageSize', 'pageToken'], relative_path='v1/{+source}:fetchInventory', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesFetchInventoryRequest', response_type_name='FetchInventoryResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Source.

      Args:
        request: (VmmigrationProjectsLocationsSourcesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Source) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}', http_method='GET', method_id='vmmigration.projects.locations.sources.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesGetRequest', response_type_name='Source', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Sources in a given project and location.

      Args:
        request: (VmmigrationProjectsLocationsSourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources', http_method='GET', method_id='vmmigration.projects.locations.sources.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/sources', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesListRequest', response_type_name='ListSourcesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Source.

      Args:
        request: (VmmigrationProjectsLocationsSourcesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}', http_method='PATCH', method_id='vmmigration.projects.locations.sources.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='source', request_type_name='VmmigrationProjectsLocationsSourcesPatchRequest', response_type_name='Operation', supports_download=False)