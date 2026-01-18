from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
class ProjectsInstanceConfigOperationsService(base_api.BaseApiService):
    """Service class for the projects_instanceConfigOperations resource."""
    _NAME = 'projects_instanceConfigOperations'

    def __init__(self, client):
        super(SpannerV1.ProjectsInstanceConfigOperationsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists the user-managed instance config long-running operations in the given project. An instance config operation has a name of the form `projects//instanceConfigs//operations/`. The long-running operation metadata field type `metadata.type_url` describes the type of the metadata. Operations returned include those that have completed/failed/canceled within the last 7 days, and pending operations. Operations returned are ordered by `operation.metadata.value.start_time` in descending order starting from the most recently started operation.

      Args:
        request: (SpannerProjectsInstanceConfigOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListInstanceConfigOperationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instanceConfigOperations', http_method='GET', method_id='spanner.projects.instanceConfigOperations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/instanceConfigOperations', request_field='', request_type_name='SpannerProjectsInstanceConfigOperationsListRequest', response_type_name='ListInstanceConfigOperationsResponse', supports_download=False)