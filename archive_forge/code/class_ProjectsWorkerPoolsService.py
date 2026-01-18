from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1alpha2 import cloudbuild_v1alpha2_messages as messages
class ProjectsWorkerPoolsService(base_api.BaseApiService):
    """Service class for the projects_workerPools resource."""
    _NAME = 'projects_workerPools'

    def __init__(self, client):
        super(CloudbuildV1alpha2.ProjectsWorkerPoolsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a `WorkerPool` to run the builds, and returns the new worker pool.

      Args:
        request: (CloudbuildProjectsWorkerPoolsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkerPool) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/workerPools', http_method='POST', method_id='cloudbuild.projects.workerPools.create', ordered_params=['parent'], path_params=['parent'], query_params=['workerPoolId'], relative_path='v1alpha2/{+parent}/workerPools', request_field='workerPool', request_type_name='CloudbuildProjectsWorkerPoolsCreateRequest', response_type_name='WorkerPool', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a `WorkerPool`.

      Args:
        request: (CloudbuildProjectsWorkerPoolsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/workerPools/{workerPoolsId}', http_method='DELETE', method_id='cloudbuild.projects.workerPools.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='CloudbuildProjectsWorkerPoolsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns details of a `WorkerPool`.

      Args:
        request: (CloudbuildProjectsWorkerPoolsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkerPool) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/workerPools/{workerPoolsId}', http_method='GET', method_id='cloudbuild.projects.workerPools.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='CloudbuildProjectsWorkerPoolsGetRequest', response_type_name='WorkerPool', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `WorkerPool`s by project.

      Args:
        request: (CloudbuildProjectsWorkerPoolsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkerPoolsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/workerPools', http_method='GET', method_id='cloudbuild.projects.workerPools.list', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha2/{+parent}/workerPools', request_field='', request_type_name='CloudbuildProjectsWorkerPoolsListRequest', response_type_name='ListWorkerPoolsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a `WorkerPool`.

      Args:
        request: (CloudbuildProjectsWorkerPoolsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkerPool) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/workerPools/{workerPoolsId}', http_method='PATCH', method_id='cloudbuild.projects.workerPools.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha2/{+name}', request_field='workerPool', request_type_name='CloudbuildProjectsWorkerPoolsPatchRequest', response_type_name='WorkerPool', supports_download=False)