from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datapipelines.v1 import datapipelines_v1_messages as messages
class ProjectsLocationsPipelinesService(base_api.BaseApiService):
    """Service class for the projects_locations_pipelines resource."""
    _NAME = 'projects_locations_pipelines'

    def __init__(self, client):
        super(DatapipelinesV1.ProjectsLocationsPipelinesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a pipeline. For a batch pipeline, you can pass scheduler information. Data Pipelines uses the scheduler information to create an internal scheduler that runs jobs periodically. If the internal scheduler is not configured, you can use RunPipeline to run jobs.

      Args:
        request: (DatapipelinesProjectsLocationsPipelinesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatapipelinesV1Pipeline) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/pipelines', http_method='POST', method_id='datapipelines.projects.locations.pipelines.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/pipelines', request_field='googleCloudDatapipelinesV1Pipeline', request_type_name='DatapipelinesProjectsLocationsPipelinesCreateRequest', response_type_name='GoogleCloudDatapipelinesV1Pipeline', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a pipeline. If a scheduler job is attached to the pipeline, it will be deleted.

      Args:
        request: (DatapipelinesProjectsLocationsPipelinesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/pipelines/{pipelinesId}', http_method='DELETE', method_id='datapipelines.projects.locations.pipelines.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DatapipelinesProjectsLocationsPipelinesDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Looks up a single pipeline. Returns a "NOT_FOUND" error if no such pipeline exists. Returns a "FORBIDDEN" error if the caller doesn't have permission to access it.

      Args:
        request: (DatapipelinesProjectsLocationsPipelinesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatapipelinesV1Pipeline) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/pipelines/{pipelinesId}', http_method='GET', method_id='datapipelines.projects.locations.pipelines.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DatapipelinesProjectsLocationsPipelinesGetRequest', response_type_name='GoogleCloudDatapipelinesV1Pipeline', supports_download=False)

    def List(self, request, global_params=None):
        """Lists pipelines. Returns a "FORBIDDEN" error if the caller doesn't have permission to access it.

      Args:
        request: (DatapipelinesProjectsLocationsPipelinesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatapipelinesV1ListPipelinesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/pipelines', http_method='GET', method_id='datapipelines.projects.locations.pipelines.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/pipelines', request_field='', request_type_name='DatapipelinesProjectsLocationsPipelinesListRequest', response_type_name='GoogleCloudDatapipelinesV1ListPipelinesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a pipeline. If successful, the updated Pipeline is returned. Returns `NOT_FOUND` if the pipeline doesn't exist. If UpdatePipeline does not return successfully, you can retry the UpdatePipeline request until you receive a successful response.

      Args:
        request: (DatapipelinesProjectsLocationsPipelinesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatapipelinesV1Pipeline) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/pipelines/{pipelinesId}', http_method='PATCH', method_id='datapipelines.projects.locations.pipelines.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudDatapipelinesV1Pipeline', request_type_name='DatapipelinesProjectsLocationsPipelinesPatchRequest', response_type_name='GoogleCloudDatapipelinesV1Pipeline', supports_download=False)

    def Run(self, request, global_params=None):
        """Creates a job for the specified pipeline directly. You can use this method when the internal scheduler is not configured and you want to trigger the job directly or through an external system. Returns a "NOT_FOUND" error if the pipeline doesn't exist. Returns a "FORBIDDEN" error if the user doesn't have permission to access the pipeline or run jobs for the pipeline.

      Args:
        request: (DatapipelinesProjectsLocationsPipelinesRunRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatapipelinesV1RunPipelineResponse) The response message.
      """
        config = self.GetMethodConfig('Run')
        return self._RunMethod(config, request, global_params=global_params)
    Run.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/pipelines/{pipelinesId}:run', http_method='POST', method_id='datapipelines.projects.locations.pipelines.run', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:run', request_field='googleCloudDatapipelinesV1RunPipelineRequest', request_type_name='DatapipelinesProjectsLocationsPipelinesRunRequest', response_type_name='GoogleCloudDatapipelinesV1RunPipelineResponse', supports_download=False)

    def Stop(self, request, global_params=None):
        """Freezes pipeline execution permanently. If there's a corresponding scheduler entry, it's deleted, and the pipeline state is changed to "ARCHIVED". However, pipeline metadata is retained.

      Args:
        request: (DatapipelinesProjectsLocationsPipelinesStopRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatapipelinesV1Pipeline) The response message.
      """
        config = self.GetMethodConfig('Stop')
        return self._RunMethod(config, request, global_params=global_params)
    Stop.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/pipelines/{pipelinesId}:stop', http_method='POST', method_id='datapipelines.projects.locations.pipelines.stop', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:stop', request_field='googleCloudDatapipelinesV1StopPipelineRequest', request_type_name='DatapipelinesProjectsLocationsPipelinesStopRequest', response_type_name='GoogleCloudDatapipelinesV1Pipeline', supports_download=False)