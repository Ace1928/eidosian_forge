from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.documentai.v1 import documentai_v1_messages as messages
class ProjectsLocationsProcessorTypesService(base_api.BaseApiService):
    """Service class for the projects_locations_processorTypes resource."""
    _NAME = 'projects_locations_processorTypes'

    def __init__(self, client):
        super(DocumentaiV1.ProjectsLocationsProcessorTypesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a processor type detail.

      Args:
        request: (DocumentaiProjectsLocationsProcessorTypesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDocumentaiV1ProcessorType) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processorTypes/{processorTypesId}', http_method='GET', method_id='documentai.projects.locations.processorTypes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DocumentaiProjectsLocationsProcessorTypesGetRequest', response_type_name='GoogleCloudDocumentaiV1ProcessorType', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the processor types that exist.

      Args:
        request: (DocumentaiProjectsLocationsProcessorTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDocumentaiV1ListProcessorTypesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processorTypes', http_method='GET', method_id='documentai.projects.locations.processorTypes.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/processorTypes', request_field='', request_type_name='DocumentaiProjectsLocationsProcessorTypesListRequest', response_type_name='GoogleCloudDocumentaiV1ListProcessorTypesResponse', supports_download=False)