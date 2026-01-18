from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataplex.v1 import dataplex_v1_messages as messages
class ProjectsLocationsLakesZonesEntitiesPartitionsService(base_api.BaseApiService):
    """Service class for the projects_locations_lakes_zones_entities_partitions resource."""
    _NAME = 'projects_locations_lakes_zones_entities_partitions'

    def __init__(self, client):
        super(DataplexV1.ProjectsLocationsLakesZonesEntitiesPartitionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a metadata partition.

      Args:
        request: (DataplexProjectsLocationsLakesZonesEntitiesPartitionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1Partition) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/zones/{zonesId}/entities/{entitiesId}/partitions', http_method='POST', method_id='dataplex.projects.locations.lakes.zones.entities.partitions.create', ordered_params=['parent'], path_params=['parent'], query_params=['validateOnly'], relative_path='v1/{+parent}/partitions', request_field='googleCloudDataplexV1Partition', request_type_name='DataplexProjectsLocationsLakesZonesEntitiesPartitionsCreateRequest', response_type_name='GoogleCloudDataplexV1Partition', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete a metadata partition.

      Args:
        request: (DataplexProjectsLocationsLakesZonesEntitiesPartitionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/zones/{zonesId}/entities/{entitiesId}/partitions/{partitionsId}', http_method='DELETE', method_id='dataplex.projects.locations.lakes.zones.entities.partitions.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v1/{+name}', request_field='', request_type_name='DataplexProjectsLocationsLakesZonesEntitiesPartitionsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Get a metadata partition of an entity.

      Args:
        request: (DataplexProjectsLocationsLakesZonesEntitiesPartitionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1Partition) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/zones/{zonesId}/entities/{entitiesId}/partitions/{partitionsId}', http_method='GET', method_id='dataplex.projects.locations.lakes.zones.entities.partitions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DataplexProjectsLocationsLakesZonesEntitiesPartitionsGetRequest', response_type_name='GoogleCloudDataplexV1Partition', supports_download=False)

    def List(self, request, global_params=None):
        """List metadata partitions of an entity.

      Args:
        request: (DataplexProjectsLocationsLakesZonesEntitiesPartitionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1ListPartitionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/zones/{zonesId}/entities/{entitiesId}/partitions', http_method='GET', method_id='dataplex.projects.locations.lakes.zones.entities.partitions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/partitions', request_field='', request_type_name='DataplexProjectsLocationsLakesZonesEntitiesPartitionsListRequest', response_type_name='GoogleCloudDataplexV1ListPartitionsResponse', supports_download=False)