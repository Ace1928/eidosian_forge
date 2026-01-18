from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1 import datacatalog_v1_messages as messages
class ProjectsLocationsTagTemplatesFieldsService(base_api.BaseApiService):
    """Service class for the projects_locations_tagTemplates_fields resource."""
    _NAME = 'projects_locations_tagTemplates_fields'

    def __init__(self, client):
        super(DatacatalogV1.ProjectsLocationsTagTemplatesFieldsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a field in a tag template. You must enable the Data Catalog API in the project identified by the `parent` parameter. For more information, see [Data Catalog resource project](https://cloud.google.com/data-catalog/docs/concepts/resource-project).

      Args:
        request: (DatacatalogProjectsLocationsTagTemplatesFieldsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1TagTemplateField) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tagTemplates/{tagTemplatesId}/fields', http_method='POST', method_id='datacatalog.projects.locations.tagTemplates.fields.create', ordered_params=['parent'], path_params=['parent'], query_params=['tagTemplateFieldId'], relative_path='v1/{+parent}/fields', request_field='googleCloudDatacatalogV1TagTemplateField', request_type_name='DatacatalogProjectsLocationsTagTemplatesFieldsCreateRequest', response_type_name='GoogleCloudDatacatalogV1TagTemplateField', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a field in a tag template and all uses of this field from the tags based on this template. You must enable the Data Catalog API in the project identified by the `name` parameter. For more information, see [Data Catalog resource project](https://cloud.google.com/data-catalog/docs/concepts/resource-project).

      Args:
        request: (DatacatalogProjectsLocationsTagTemplatesFieldsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tagTemplates/{tagTemplatesId}/fields/{fieldsId}', http_method='DELETE', method_id='datacatalog.projects.locations.tagTemplates.fields.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v1/{+name}', request_field='', request_type_name='DatacatalogProjectsLocationsTagTemplatesFieldsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a field in a tag template. You can't update the field type with this method. You must enable the Data Catalog API in the project identified by the `name` parameter. For more information, see [Data Catalog resource project](https://cloud.google.com/data-catalog/docs/concepts/resource-project).

      Args:
        request: (DatacatalogProjectsLocationsTagTemplatesFieldsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1TagTemplateField) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tagTemplates/{tagTemplatesId}/fields/{fieldsId}', http_method='PATCH', method_id='datacatalog.projects.locations.tagTemplates.fields.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudDatacatalogV1TagTemplateField', request_type_name='DatacatalogProjectsLocationsTagTemplatesFieldsPatchRequest', response_type_name='GoogleCloudDatacatalogV1TagTemplateField', supports_download=False)

    def Rename(self, request, global_params=None):
        """Renames a field in a tag template. You must enable the Data Catalog API in the project identified by the `name` parameter. For more information, see [Data Catalog resource project] (https://cloud.google.com/data-catalog/docs/concepts/resource-project).

      Args:
        request: (DatacatalogProjectsLocationsTagTemplatesFieldsRenameRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1TagTemplateField) The response message.
      """
        config = self.GetMethodConfig('Rename')
        return self._RunMethod(config, request, global_params=global_params)
    Rename.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tagTemplates/{tagTemplatesId}/fields/{fieldsId}:rename', http_method='POST', method_id='datacatalog.projects.locations.tagTemplates.fields.rename', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:rename', request_field='googleCloudDatacatalogV1RenameTagTemplateFieldRequest', request_type_name='DatacatalogProjectsLocationsTagTemplatesFieldsRenameRequest', response_type_name='GoogleCloudDatacatalogV1TagTemplateField', supports_download=False)