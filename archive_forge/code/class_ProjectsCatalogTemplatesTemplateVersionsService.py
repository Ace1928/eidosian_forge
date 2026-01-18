from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
class ProjectsCatalogTemplatesTemplateVersionsService(base_api.BaseApiService):
    """Service class for the projects_catalogTemplates_templateVersions resource."""
    _NAME = 'projects_catalogTemplates_templateVersions'

    def __init__(self, client):
        super(DataflowV1b3.ProjectsCatalogTemplatesTemplateVersionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Template with TemplateVersion. Requires project_id(projects) and template display_name(catalogTemplates). The template display_name is set by the user.

      Args:
        request: (DataflowProjectsCatalogTemplatesTemplateVersionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TemplateVersion) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1b3/projects/{projectsId}/catalogTemplates/{catalogTemplatesId}/templateVersions', http_method='POST', method_id='dataflow.projects.catalogTemplates.templateVersions.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1b3/{+parent}/templateVersions', request_field='createTemplateVersionRequest', request_type_name='DataflowProjectsCatalogTemplatesTemplateVersionsCreateRequest', response_type_name='TemplateVersion', supports_download=False)