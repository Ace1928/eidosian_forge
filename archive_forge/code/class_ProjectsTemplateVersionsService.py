from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
class ProjectsTemplateVersionsService(base_api.BaseApiService):
    """Service class for the projects_templateVersions resource."""
    _NAME = 'projects_templateVersions'

    def __init__(self, client):
        super(DataflowV1b3.ProjectsTemplateVersionsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """List TemplateVersions using project_id and an optional display_name field. List all the TemplateVersions in the Template if display set. List all the TemplateVersions in the Project if display_name not set.

      Args:
        request: (DataflowProjectsTemplateVersionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTemplateVersionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1b3/projects/{projectsId}/templateVersions', http_method='GET', method_id='dataflow.projects.templateVersions.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1b3/{+parent}/templateVersions', request_field='', request_type_name='DataflowProjectsTemplateVersionsListRequest', response_type_name='ListTemplateVersionsResponse', supports_download=False)