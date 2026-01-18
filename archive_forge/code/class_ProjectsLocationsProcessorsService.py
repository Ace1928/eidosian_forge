from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.documentai.v1 import documentai_v1_messages as messages
class ProjectsLocationsProcessorsService(base_api.BaseApiService):
    """Service class for the projects_locations_processors resource."""
    _NAME = 'projects_locations_processors'

    def __init__(self, client):
        super(DocumentaiV1.ProjectsLocationsProcessorsService, self).__init__(client)
        self._upload_configs = {}

    def BatchProcess(self, request, global_params=None):
        """LRO endpoint to batch process many documents. The output is written to Cloud Storage as JSON in the [Document] format.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsBatchProcessRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('BatchProcess')
        return self._RunMethod(config, request, global_params=global_params)
    BatchProcess.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}:batchProcess', http_method='POST', method_id='documentai.projects.locations.processors.batchProcess', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:batchProcess', request_field='googleCloudDocumentaiV1BatchProcessRequest', request_type_name='DocumentaiProjectsLocationsProcessorsBatchProcessRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a processor from the ProcessorType provided. The processor will be at `ENABLED` state by default after its creation.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDocumentaiV1Processor) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors', http_method='POST', method_id='documentai.projects.locations.processors.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/processors', request_field='googleCloudDocumentaiV1Processor', request_type_name='DocumentaiProjectsLocationsProcessorsCreateRequest', response_type_name='GoogleCloudDocumentaiV1Processor', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the processor, unloads all deployed model artifacts if it was enabled and then deletes all artifacts associated with this processor.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}', http_method='DELETE', method_id='documentai.projects.locations.processors.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DocumentaiProjectsLocationsProcessorsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Disable(self, request, global_params=None):
        """Disables a processor.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsDisableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Disable')
        return self._RunMethod(config, request, global_params=global_params)
    Disable.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}:disable', http_method='POST', method_id='documentai.projects.locations.processors.disable', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:disable', request_field='googleCloudDocumentaiV1DisableProcessorRequest', request_type_name='DocumentaiProjectsLocationsProcessorsDisableRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Enable(self, request, global_params=None):
        """Enables a processor.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsEnableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Enable')
        return self._RunMethod(config, request, global_params=global_params)
    Enable.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}:enable', http_method='POST', method_id='documentai.projects.locations.processors.enable', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:enable', request_field='googleCloudDocumentaiV1EnableProcessorRequest', request_type_name='DocumentaiProjectsLocationsProcessorsEnableRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a processor detail.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDocumentaiV1Processor) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}', http_method='GET', method_id='documentai.projects.locations.processors.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DocumentaiProjectsLocationsProcessorsGetRequest', response_type_name='GoogleCloudDocumentaiV1Processor', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all processors which belong to this project.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDocumentaiV1ListProcessorsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors', http_method='GET', method_id='documentai.projects.locations.processors.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/processors', request_field='', request_type_name='DocumentaiProjectsLocationsProcessorsListRequest', response_type_name='GoogleCloudDocumentaiV1ListProcessorsResponse', supports_download=False)

    def Process(self, request, global_params=None):
        """Processes a single document.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsProcessRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDocumentaiV1ProcessResponse) The response message.
      """
        config = self.GetMethodConfig('Process')
        return self._RunMethod(config, request, global_params=global_params)
    Process.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}:process', http_method='POST', method_id='documentai.projects.locations.processors.process', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:process', request_field='googleCloudDocumentaiV1ProcessRequest', request_type_name='DocumentaiProjectsLocationsProcessorsProcessRequest', response_type_name='GoogleCloudDocumentaiV1ProcessResponse', supports_download=False)

    def SetDefaultProcessorVersion(self, request, global_params=None):
        """Set the default (active) version of a Processor that will be used in ProcessDocument and BatchProcessDocuments.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsSetDefaultProcessorVersionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('SetDefaultProcessorVersion')
        return self._RunMethod(config, request, global_params=global_params)
    SetDefaultProcessorVersion.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}:setDefaultProcessorVersion', http_method='POST', method_id='documentai.projects.locations.processors.setDefaultProcessorVersion', ordered_params=['processor'], path_params=['processor'], query_params=[], relative_path='v1/{+processor}:setDefaultProcessorVersion', request_field='googleCloudDocumentaiV1SetDefaultProcessorVersionRequest', request_type_name='DocumentaiProjectsLocationsProcessorsSetDefaultProcessorVersionRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)