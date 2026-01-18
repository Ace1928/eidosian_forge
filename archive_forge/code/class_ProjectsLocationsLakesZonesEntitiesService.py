from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataplex.v1 import dataplex_v1_messages as messages
class ProjectsLocationsLakesZonesEntitiesService(base_api.BaseApiService):
    """Service class for the projects_locations_lakes_zones_entities resource."""
    _NAME = 'projects_locations_lakes_zones_entities'

    def __init__(self, client):
        super(DataplexV1.ProjectsLocationsLakesZonesEntitiesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a metadata entity.

      Args:
        request: (DataplexProjectsLocationsLakesZonesEntitiesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1Entity) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/zones/{zonesId}/entities', http_method='POST', method_id='dataplex.projects.locations.lakes.zones.entities.create', ordered_params=['parent'], path_params=['parent'], query_params=['validateOnly'], relative_path='v1/{+parent}/entities', request_field='googleCloudDataplexV1Entity', request_type_name='DataplexProjectsLocationsLakesZonesEntitiesCreateRequest', response_type_name='GoogleCloudDataplexV1Entity', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete a metadata entity.

      Args:
        request: (DataplexProjectsLocationsLakesZonesEntitiesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/zones/{zonesId}/entities/{entitiesId}', http_method='DELETE', method_id='dataplex.projects.locations.lakes.zones.entities.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v1/{+name}', request_field='', request_type_name='DataplexProjectsLocationsLakesZonesEntitiesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Get a metadata entity.

      Args:
        request: (DataplexProjectsLocationsLakesZonesEntitiesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1Entity) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/zones/{zonesId}/entities/{entitiesId}', http_method='GET', method_id='dataplex.projects.locations.lakes.zones.entities.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1/{+name}', request_field='', request_type_name='DataplexProjectsLocationsLakesZonesEntitiesGetRequest', response_type_name='GoogleCloudDataplexV1Entity', supports_download=False)

    def List(self, request, global_params=None):
        """List metadata entities in a zone.

      Args:
        request: (DataplexProjectsLocationsLakesZonesEntitiesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1ListEntitiesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/zones/{zonesId}/entities', http_method='GET', method_id='dataplex.projects.locations.lakes.zones.entities.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'view'], relative_path='v1/{+parent}/entities', request_field='', request_type_name='DataplexProjectsLocationsLakesZonesEntitiesListRequest', response_type_name='GoogleCloudDataplexV1ListEntitiesResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Update a metadata entity. Only supports full resource update.

      Args:
        request: (DataplexProjectsLocationsLakesZonesEntitiesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1Entity) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/zones/{zonesId}/entities/{entitiesId}', http_method='PUT', method_id='dataplex.projects.locations.lakes.zones.entities.update', ordered_params=['name'], path_params=['name'], query_params=['validateOnly'], relative_path='v1/{+name}', request_field='googleCloudDataplexV1Entity', request_type_name='DataplexProjectsLocationsLakesZonesEntitiesUpdateRequest', response_type_name='GoogleCloudDataplexV1Entity', supports_download=False)