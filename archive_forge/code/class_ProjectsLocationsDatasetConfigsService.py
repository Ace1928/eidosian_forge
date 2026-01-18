from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.storageinsights.v1 import storageinsights_v1_messages as messages
class ProjectsLocationsDatasetConfigsService(base_api.BaseApiService):
    """Service class for the projects_locations_datasetConfigs resource."""
    _NAME = 'projects_locations_datasetConfigs'

    def __init__(self, client):
        super(StorageinsightsV1.ProjectsLocationsDatasetConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create method for the projects_locations_datasetConfigs service.

      Args:
        request: (StorageinsightsProjectsLocationsDatasetConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasetConfigs', http_method='POST', method_id='storageinsights.projects.locations.datasetConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['datasetConfigId', 'requestId'], relative_path='v1/{+parent}/datasetConfigs', request_field='datasetConfig', request_type_name='StorageinsightsProjectsLocationsDatasetConfigsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete method for the projects_locations_datasetConfigs service.

      Args:
        request: (StorageinsightsProjectsLocationsDatasetConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasetConfigs/{datasetConfigsId}', http_method='DELETE', method_id='storageinsights.projects.locations.datasetConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='StorageinsightsProjectsLocationsDatasetConfigsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Get method for the projects_locations_datasetConfigs service.

      Args:
        request: (StorageinsightsProjectsLocationsDatasetConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DatasetConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasetConfigs/{datasetConfigsId}', http_method='GET', method_id='storageinsights.projects.locations.datasetConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='StorageinsightsProjectsLocationsDatasetConfigsGetRequest', response_type_name='DatasetConfig', supports_download=False)

    def LinkDataset(self, request, global_params=None):
        """LinkDataset method for the projects_locations_datasetConfigs service.

      Args:
        request: (StorageinsightsProjectsLocationsDatasetConfigsLinkDatasetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('LinkDataset')
        return self._RunMethod(config, request, global_params=global_params)
    LinkDataset.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasetConfigs/{datasetConfigsId}:linkDataset', http_method='POST', method_id='storageinsights.projects.locations.datasetConfigs.linkDataset', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:linkDataset', request_field='linkDatasetRequest', request_type_name='StorageinsightsProjectsLocationsDatasetConfigsLinkDatasetRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """List method for the projects_locations_datasetConfigs service.

      Args:
        request: (StorageinsightsProjectsLocationsDatasetConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDatasetConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasetConfigs', http_method='GET', method_id='storageinsights.projects.locations.datasetConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/datasetConfigs', request_field='', request_type_name='StorageinsightsProjectsLocationsDatasetConfigsListRequest', response_type_name='ListDatasetConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patch method for the projects_locations_datasetConfigs service.

      Args:
        request: (StorageinsightsProjectsLocationsDatasetConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasetConfigs/{datasetConfigsId}', http_method='PATCH', method_id='storageinsights.projects.locations.datasetConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='datasetConfig', request_type_name='StorageinsightsProjectsLocationsDatasetConfigsPatchRequest', response_type_name='Operation', supports_download=False)

    def UnlinkDataset(self, request, global_params=None):
        """UnlinkDataset method for the projects_locations_datasetConfigs service.

      Args:
        request: (StorageinsightsProjectsLocationsDatasetConfigsUnlinkDatasetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('UnlinkDataset')
        return self._RunMethod(config, request, global_params=global_params)
    UnlinkDataset.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasetConfigs/{datasetConfigsId}:unlinkDataset', http_method='POST', method_id='storageinsights.projects.locations.datasetConfigs.unlinkDataset', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:unlinkDataset', request_field='unlinkDatasetRequest', request_type_name='StorageinsightsProjectsLocationsDatasetConfigsUnlinkDatasetRequest', response_type_name='Operation', supports_download=False)