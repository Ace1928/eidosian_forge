from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.translate.v3 import translate_v3_messages as messages
class ProjectsLocationsAdaptiveMtDatasetsService(base_api.BaseApiService):
    """Service class for the projects_locations_adaptiveMtDatasets resource."""
    _NAME = 'projects_locations_adaptiveMtDatasets'

    def __init__(self, client):
        super(TranslateV3.ProjectsLocationsAdaptiveMtDatasetsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an Adaptive MT dataset.

      Args:
        request: (TranslateProjectsLocationsAdaptiveMtDatasetsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AdaptiveMtDataset) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/locations/{locationsId}/adaptiveMtDatasets', http_method='POST', method_id='translate.projects.locations.adaptiveMtDatasets.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v3/{+parent}/adaptiveMtDatasets', request_field='adaptiveMtDataset', request_type_name='TranslateProjectsLocationsAdaptiveMtDatasetsCreateRequest', response_type_name='AdaptiveMtDataset', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an Adaptive MT dataset, including all its entries and associated metadata.

      Args:
        request: (TranslateProjectsLocationsAdaptiveMtDatasetsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/locations/{locationsId}/adaptiveMtDatasets/{adaptiveMtDatasetsId}', http_method='DELETE', method_id='translate.projects.locations.adaptiveMtDatasets.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='TranslateProjectsLocationsAdaptiveMtDatasetsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the Adaptive MT dataset.

      Args:
        request: (TranslateProjectsLocationsAdaptiveMtDatasetsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AdaptiveMtDataset) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/locations/{locationsId}/adaptiveMtDatasets/{adaptiveMtDatasetsId}', http_method='GET', method_id='translate.projects.locations.adaptiveMtDatasets.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='TranslateProjectsLocationsAdaptiveMtDatasetsGetRequest', response_type_name='AdaptiveMtDataset', supports_download=False)

    def ImportAdaptiveMtFile(self, request, global_params=None):
        """Imports an AdaptiveMtFile and adds all of its sentences into the AdaptiveMtDataset.

      Args:
        request: (TranslateProjectsLocationsAdaptiveMtDatasetsImportAdaptiveMtFileRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ImportAdaptiveMtFileResponse) The response message.
      """
        config = self.GetMethodConfig('ImportAdaptiveMtFile')
        return self._RunMethod(config, request, global_params=global_params)
    ImportAdaptiveMtFile.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/locations/{locationsId}/adaptiveMtDatasets/{adaptiveMtDatasetsId}:importAdaptiveMtFile', http_method='POST', method_id='translate.projects.locations.adaptiveMtDatasets.importAdaptiveMtFile', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v3/{+parent}:importAdaptiveMtFile', request_field='importAdaptiveMtFileRequest', request_type_name='TranslateProjectsLocationsAdaptiveMtDatasetsImportAdaptiveMtFileRequest', response_type_name='ImportAdaptiveMtFileResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all Adaptive MT datasets for which the caller has read permission.

      Args:
        request: (TranslateProjectsLocationsAdaptiveMtDatasetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAdaptiveMtDatasetsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/locations/{locationsId}/adaptiveMtDatasets', http_method='GET', method_id='translate.projects.locations.adaptiveMtDatasets.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v3/{+parent}/adaptiveMtDatasets', request_field='', request_type_name='TranslateProjectsLocationsAdaptiveMtDatasetsListRequest', response_type_name='ListAdaptiveMtDatasetsResponse', supports_download=False)