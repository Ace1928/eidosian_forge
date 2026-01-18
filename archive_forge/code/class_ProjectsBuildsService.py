from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class ProjectsBuildsService(base_api.BaseApiService):
    """Service class for the projects_builds resource."""
    _NAME = 'projects_builds'

    def __init__(self, client):
        super(CloudbuildV1.ProjectsBuildsService, self).__init__(client)
        self._upload_configs = {}

    def Approve(self, request, global_params=None):
        """Approves or rejects a pending build. If approved, the returned LRO will be analogous to the LRO returned from a CreateBuild call. If rejected, the returned LRO will be immediately done.

      Args:
        request: (CloudbuildProjectsBuildsApproveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Approve')
        return self._RunMethod(config, request, global_params=global_params)
    Approve.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/builds/{buildsId}:approve', http_method='POST', method_id='cloudbuild.projects.builds.approve', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:approve', request_field='approveBuildRequest', request_type_name='CloudbuildProjectsBuildsApproveRequest', response_type_name='Operation', supports_download=False)

    def Cancel(self, request, global_params=None):
        """Cancels a build in progress.

      Args:
        request: (CloudbuildProjectsBuildsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Build) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='cloudbuild.projects.builds.cancel', ordered_params=['projectId', 'id'], path_params=['id', 'projectId'], query_params=[], relative_path='v1/projects/{projectId}/builds/{id}:cancel', request_field='cancelBuildRequest', request_type_name='CloudbuildProjectsBuildsCancelRequest', response_type_name='Build', supports_download=False)

    def Create(self, request, global_params=None):
        """Starts a build with the specified configuration. This method returns a long-running `Operation`, which includes the build ID. Pass the build ID to `GetBuild` to determine the build status (such as `SUCCESS` or `FAILURE`).

      Args:
        request: (CloudbuildProjectsBuildsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='cloudbuild.projects.builds.create', ordered_params=['projectId'], path_params=['projectId'], query_params=['parent'], relative_path='v1/projects/{projectId}/builds', request_field='build', request_type_name='CloudbuildProjectsBuildsCreateRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns information about a previously requested build. The `Build` that is returned includes its status (such as `SUCCESS`, `FAILURE`, or `WORKING`), and timing information.

      Args:
        request: (CloudbuildProjectsBuildsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Build) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudbuild.projects.builds.get', ordered_params=['projectId', 'id'], path_params=['id', 'projectId'], query_params=['name'], relative_path='v1/projects/{projectId}/builds/{id}', request_field='', request_type_name='CloudbuildProjectsBuildsGetRequest', response_type_name='Build', supports_download=False)

    def List(self, request, global_params=None):
        """Lists previously requested builds. Previously requested builds may still be in-progress, or may have finished successfully or unsuccessfully.

      Args:
        request: (CloudbuildProjectsBuildsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBuildsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudbuild.projects.builds.list', ordered_params=['projectId'], path_params=['projectId'], query_params=['filter', 'pageSize', 'pageToken', 'parent'], relative_path='v1/projects/{projectId}/builds', request_field='', request_type_name='CloudbuildProjectsBuildsListRequest', response_type_name='ListBuildsResponse', supports_download=False)

    def Retry(self, request, global_params=None):
        """Creates a new build based on the specified build. This method creates a new build using the original build request, which may or may not result in an identical build. For triggered builds: * Triggered builds resolve to a precise revision; therefore a retry of a triggered build will result in a build that uses the same revision. For non-triggered builds that specify `RepoSource`: * If the original build built from the tip of a branch, the retried build will build from the tip of that branch, which may not be the same revision as the original build. * If the original build specified a commit sha or revision ID, the retried build will use the identical source. For builds that specify `StorageSource`: * If the original build pulled source from Cloud Storage without specifying the generation of the object, the new build will use the current object, which may be different from the original build source. * If the original build pulled source from Cloud Storage and specified the generation of the object, the new build will attempt to use the same object, which may or may not be available depending on the bucket's lifecycle management settings.

      Args:
        request: (RetryBuildRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Retry')
        return self._RunMethod(config, request, global_params=global_params)
    Retry.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='cloudbuild.projects.builds.retry', ordered_params=['projectId', 'id'], path_params=['id', 'projectId'], query_params=[], relative_path='v1/projects/{projectId}/builds/{id}:retry', request_field='<request>', request_type_name='RetryBuildRequest', response_type_name='Operation', supports_download=False)