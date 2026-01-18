from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
class ProjectsCatalogTemplatesService(base_api.BaseApiService):
    """Service class for the projects_catalogTemplates resource."""
    _NAME = 'projects_catalogTemplates'

    def __init__(self, client):
        super(DataflowV1b3.ProjectsCatalogTemplatesService, self).__init__(client)
        self._upload_configs = {}

    def Commit(self, request, global_params=None):
        """Creates a new TemplateVersion (Important: not new Template) entry in the spanner table. Requires project_id and display_name (template).

      Args:
        request: (DataflowProjectsCatalogTemplatesCommitRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TemplateVersion) The response message.
      """
        config = self.GetMethodConfig('Commit')
        return self._RunMethod(config, request, global_params=global_params)
    Commit.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1b3/projects/{projectsId}/catalogTemplates/{catalogTemplatesId}:commit', http_method='POST', method_id='dataflow.projects.catalogTemplates.commit', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1b3/{+name}:commit', request_field='commitTemplateVersionRequest', request_type_name='DataflowProjectsCatalogTemplatesCommitRequest', response_type_name='TemplateVersion', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an existing Template. Do nothing if Template does not exist.

      Args:
        request: (DataflowProjectsCatalogTemplatesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1b3/projects/{projectsId}/catalogTemplates/{catalogTemplatesId}', http_method='DELETE', method_id='dataflow.projects.catalogTemplates.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1b3/{+name}', request_field='', request_type_name='DataflowProjectsCatalogTemplatesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Get TemplateVersion using project_id and display_name with an optional version_id field. Get latest (has tag "latest") TemplateVersion if version_id not set.

      Args:
        request: (DataflowProjectsCatalogTemplatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TemplateVersion) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1b3/projects/{projectsId}/catalogTemplates/{catalogTemplatesId}', http_method='GET', method_id='dataflow.projects.catalogTemplates.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1b3/{+name}', request_field='', request_type_name='DataflowProjectsCatalogTemplatesGetRequest', response_type_name='TemplateVersion', supports_download=False)

    def Label(self, request, global_params=None):
        """Updates the label of the TemplateVersion. Label can be duplicated in Template, so either add or remove the label in the TemplateVersion.

      Args:
        request: (DataflowProjectsCatalogTemplatesLabelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ModifyTemplateVersionLabelResponse) The response message.
      """
        config = self.GetMethodConfig('Label')
        return self._RunMethod(config, request, global_params=global_params)
    Label.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1b3/projects/{projectsId}/catalogTemplates/{catalogTemplatesId}:label', http_method='POST', method_id='dataflow.projects.catalogTemplates.label', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1b3/{+name}:label', request_field='modifyTemplateVersionLabelRequest', request_type_name='DataflowProjectsCatalogTemplatesLabelRequest', response_type_name='ModifyTemplateVersionLabelResponse', supports_download=False)

    def Tag(self, request, global_params=None):
        """Updates the tag of the TemplateVersion, and tag is unique in Template. If tag exists in another TemplateVersion in the Template, updates the tag to this TemplateVersion will remove it from the old TemplateVersion and add it to this TemplateVersion. If request is remove_only (remove_only = true), remove the tag from this TemplateVersion.

      Args:
        request: (DataflowProjectsCatalogTemplatesTagRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ModifyTemplateVersionTagResponse) The response message.
      """
        config = self.GetMethodConfig('Tag')
        return self._RunMethod(config, request, global_params=global_params)
    Tag.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1b3/projects/{projectsId}/catalogTemplates/{catalogTemplatesId}:tag', http_method='POST', method_id='dataflow.projects.catalogTemplates.tag', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1b3/{+name}:tag', request_field='modifyTemplateVersionTagRequest', request_type_name='DataflowProjectsCatalogTemplatesTagRequest', response_type_name='ModifyTemplateVersionTagResponse', supports_download=False)