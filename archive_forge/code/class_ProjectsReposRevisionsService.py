from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.source.v1 import source_v1_messages as messages
class ProjectsReposRevisionsService(base_api.BaseApiService):
    """Service class for the projects_repos_revisions resource."""
    _NAME = 'projects_repos_revisions'

    def __init__(self, client):
        super(SourceV1.ProjectsReposRevisionsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves revision metadata for a single revision.

      Args:
        request: (SourceProjectsReposRevisionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Revision) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='source.projects.repos.revisions.get', ordered_params=['projectId', 'repoName', 'revisionId'], path_params=['projectId', 'repoName', 'revisionId'], query_params=['repoId_uid'], relative_path='v1/projects/{projectId}/repos/{repoName}/revisions/{revisionId}', request_field='', request_type_name='SourceProjectsReposRevisionsGetRequest', response_type_name='Revision', supports_download=False)

    def GetBatchGet(self, request, global_params=None):
        """Retrieves revision metadata for several revisions at once. It returns an.
error if any retrieval fails.

      Args:
        request: (SourceProjectsReposRevisionsGetBatchGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('GetBatchGet')
        return self._RunMethod(config, request, global_params=global_params)
    GetBatchGet.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='source.projects.repos.revisions.getBatchGet', ordered_params=['projectId', 'repoName'], path_params=['projectId', 'repoName'], query_params=['repoId_uid', 'revisionIds'], relative_path='v1/projects/{projectId}/repos/{repoName}/revisions:batchGet', request_field='', request_type_name='SourceProjectsReposRevisionsGetBatchGetRequest', response_type_name='GetRevisionsResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves all revisions topologically between the starts and ends.
Uses the commit date to break ties in the topology (e.g. when a revision
has two parents).

      Args:
        request: (SourceProjectsReposRevisionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='source.projects.repos.revisions.list', ordered_params=['projectId', 'repoName'], path_params=['projectId', 'repoName'], query_params=['ends', 'pageSize', 'pageToken', 'path', 'repoId_uid', 'starts', 'walkDirection'], relative_path='v1/projects/{projectId}/repos/{repoName}/revisions', request_field='', request_type_name='SourceProjectsReposRevisionsListRequest', response_type_name='ListRevisionsResponse', supports_download=False)

    def ListFiles(self, request, global_params=None):
        """ListFiles returns a list of all files in a SourceContext. The.
information about each file includes its path and its hash.
The result is ordered by path. Pagination is supported.

      Args:
        request: (SourceProjectsReposRevisionsListFilesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFilesResponse) The response message.
      """
        config = self.GetMethodConfig('ListFiles')
        return self._RunMethod(config, request, global_params=global_params)
    ListFiles.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='source.projects.repos.revisions.listFiles', ordered_params=['projectId', 'repoName', 'revisionId'], path_params=['projectId', 'repoName', 'revisionId'], query_params=['pageSize', 'pageToken', 'sourceContext_cloudRepo_aliasContext_kind', 'sourceContext_cloudRepo_aliasContext_name', 'sourceContext_cloudRepo_aliasName', 'sourceContext_cloudRepo_repoId_uid', 'sourceContext_cloudWorkspace_snapshotId', 'sourceContext_cloudWorkspace_workspaceId_name', 'sourceContext_cloudWorkspace_workspaceId_repoId_projectRepoId_projectId', 'sourceContext_cloudWorkspace_workspaceId_repoId_projectRepoId_repoName', 'sourceContext_cloudWorkspace_workspaceId_repoId_uid', 'sourceContext_gerrit_aliasContext_kind', 'sourceContext_gerrit_aliasContext_name', 'sourceContext_gerrit_aliasName', 'sourceContext_gerrit_gerritProject', 'sourceContext_gerrit_hostUri', 'sourceContext_gerrit_revisionId', 'sourceContext_git_revisionId', 'sourceContext_git_url'], relative_path='v1/projects/{projectId}/repos/{repoName}/revisions/{revisionId}:listFiles', request_field='', request_type_name='SourceProjectsReposRevisionsListFilesRequest', response_type_name='ListFilesResponse', supports_download=False)