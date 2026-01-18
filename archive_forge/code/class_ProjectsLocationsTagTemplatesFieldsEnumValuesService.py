from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1 import datacatalog_v1_messages as messages
class ProjectsLocationsTagTemplatesFieldsEnumValuesService(base_api.BaseApiService):
    """Service class for the projects_locations_tagTemplates_fields_enumValues resource."""
    _NAME = 'projects_locations_tagTemplates_fields_enumValues'

    def __init__(self, client):
        super(DatacatalogV1.ProjectsLocationsTagTemplatesFieldsEnumValuesService, self).__init__(client)
        self._upload_configs = {}

    def Rename(self, request, global_params=None):
        """Renames an enum value in a tag template. Within a single enum field, enum values must be unique.

      Args:
        request: (DatacatalogProjectsLocationsTagTemplatesFieldsEnumValuesRenameRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1TagTemplateField) The response message.
      """
        config = self.GetMethodConfig('Rename')
        return self._RunMethod(config, request, global_params=global_params)
    Rename.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tagTemplates/{tagTemplatesId}/fields/{fieldsId}/enumValues/{enumValuesId}:rename', http_method='POST', method_id='datacatalog.projects.locations.tagTemplates.fields.enumValues.rename', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:rename', request_field='googleCloudDatacatalogV1RenameTagTemplateFieldEnumValueRequest', request_type_name='DatacatalogProjectsLocationsTagTemplatesFieldsEnumValuesRenameRequest', response_type_name='GoogleCloudDatacatalogV1TagTemplateField', supports_download=False)