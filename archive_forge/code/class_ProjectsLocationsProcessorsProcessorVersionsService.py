from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.documentai.v1 import documentai_v1_messages as messages
class ProjectsLocationsProcessorsProcessorVersionsService(base_api.BaseApiService):
    """Service class for the projects_locations_processors_processorVersions resource."""
    _NAME = 'projects_locations_processors_processorVersions'

    def __init__(self, client):
        super(DocumentaiV1.ProjectsLocationsProcessorsProcessorVersionsService, self).__init__(client)
        self._upload_configs = {}

    def BatchProcess(self, request, global_params=None):
        """LRO endpoint to batch process many documents. The output is written to Cloud Storage as JSON in the [Document] format.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsProcessorVersionsBatchProcessRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('BatchProcess')
        return self._RunMethod(config, request, global_params=global_params)
    BatchProcess.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}/processorVersions/{processorVersionsId}:batchProcess', http_method='POST', method_id='documentai.projects.locations.processors.processorVersions.batchProcess', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:batchProcess', request_field='googleCloudDocumentaiV1BatchProcessRequest', request_type_name='DocumentaiProjectsLocationsProcessorsProcessorVersionsBatchProcessRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the processor version, all artifacts under the processor version will be deleted.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsProcessorVersionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}/processorVersions/{processorVersionsId}', http_method='DELETE', method_id='documentai.projects.locations.processors.processorVersions.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DocumentaiProjectsLocationsProcessorsProcessorVersionsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Deploy(self, request, global_params=None):
        """Deploys the processor version.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsProcessorVersionsDeployRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Deploy')
        return self._RunMethod(config, request, global_params=global_params)
    Deploy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}/processorVersions/{processorVersionsId}:deploy', http_method='POST', method_id='documentai.projects.locations.processors.processorVersions.deploy', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:deploy', request_field='googleCloudDocumentaiV1DeployProcessorVersionRequest', request_type_name='DocumentaiProjectsLocationsProcessorsProcessorVersionsDeployRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def EvaluateProcessorVersion(self, request, global_params=None):
        """Evaluates a ProcessorVersion against annotated documents, producing an Evaluation.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsProcessorVersionsEvaluateProcessorVersionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('EvaluateProcessorVersion')
        return self._RunMethod(config, request, global_params=global_params)
    EvaluateProcessorVersion.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}/processorVersions/{processorVersionsId}:evaluateProcessorVersion', http_method='POST', method_id='documentai.projects.locations.processors.processorVersions.evaluateProcessorVersion', ordered_params=['processorVersion'], path_params=['processorVersion'], query_params=[], relative_path='v1/{+processorVersion}:evaluateProcessorVersion', request_field='googleCloudDocumentaiV1EvaluateProcessorVersionRequest', request_type_name='DocumentaiProjectsLocationsProcessorsProcessorVersionsEvaluateProcessorVersionRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a processor version detail.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsProcessorVersionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDocumentaiV1ProcessorVersion) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}/processorVersions/{processorVersionsId}', http_method='GET', method_id='documentai.projects.locations.processors.processorVersions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DocumentaiProjectsLocationsProcessorsProcessorVersionsGetRequest', response_type_name='GoogleCloudDocumentaiV1ProcessorVersion', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all versions of a processor.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsProcessorVersionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDocumentaiV1ListProcessorVersionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}/processorVersions', http_method='GET', method_id='documentai.projects.locations.processors.processorVersions.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/processorVersions', request_field='', request_type_name='DocumentaiProjectsLocationsProcessorsProcessorVersionsListRequest', response_type_name='GoogleCloudDocumentaiV1ListProcessorVersionsResponse', supports_download=False)

    def Process(self, request, global_params=None):
        """Processes a single document.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsProcessorVersionsProcessRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDocumentaiV1ProcessResponse) The response message.
      """
        config = self.GetMethodConfig('Process')
        return self._RunMethod(config, request, global_params=global_params)
    Process.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}/processorVersions/{processorVersionsId}:process', http_method='POST', method_id='documentai.projects.locations.processors.processorVersions.process', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:process', request_field='googleCloudDocumentaiV1ProcessRequest', request_type_name='DocumentaiProjectsLocationsProcessorsProcessorVersionsProcessRequest', response_type_name='GoogleCloudDocumentaiV1ProcessResponse', supports_download=False)

    def Train(self, request, global_params=None):
        """Trains a new processor version. Operation metadata is returned as TrainProcessorVersionMetadata.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsProcessorVersionsTrainRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Train')
        return self._RunMethod(config, request, global_params=global_params)
    Train.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}/processorVersions:train', http_method='POST', method_id='documentai.projects.locations.processors.processorVersions.train', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/processorVersions:train', request_field='googleCloudDocumentaiV1TrainProcessorVersionRequest', request_type_name='DocumentaiProjectsLocationsProcessorsProcessorVersionsTrainRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Undeploy(self, request, global_params=None):
        """Undeploys the processor version.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsProcessorVersionsUndeployRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Undeploy')
        return self._RunMethod(config, request, global_params=global_params)
    Undeploy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}/processorVersions/{processorVersionsId}:undeploy', http_method='POST', method_id='documentai.projects.locations.processors.processorVersions.undeploy', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:undeploy', request_field='googleCloudDocumentaiV1UndeployProcessorVersionRequest', request_type_name='DocumentaiProjectsLocationsProcessorsProcessorVersionsUndeployRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)