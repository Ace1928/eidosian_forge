from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firebasedataconnect.v1alpha import firebasedataconnect_v1alpha_messages as messages
class ProjectsLocationsServicesSchemasService(base_api.BaseApiService):
    """Service class for the projects_locations_services_schemas resource."""
    _NAME = 'projects_locations_services_schemas'

    def __init__(self, client):
        super(FirebasedataconnectV1alpha.ProjectsLocationsServicesSchemasService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Schema in a given project and location. Only creation of `schemas/main` is supported and calling create with any other schema ID will result in an error.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesSchemasCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/schemas', http_method='POST', method_id='firebasedataconnect.projects.locations.services.schemas.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'schemaId', 'validateOnly'], relative_path='v1alpha/{+parent}/schemas', request_field='schema', request_type_name='FirebasedataconnectProjectsLocationsServicesSchemasCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Schema. Because the schema and connectors must be compatible at all times, if this is called while any connectors are active, this will result in an error.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesSchemasDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/schemas/{schemasId}', http_method='DELETE', method_id='firebasedataconnect.projects.locations.services.schemas.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'force', 'requestId', 'validateOnly'], relative_path='v1alpha/{+name}', request_field='', request_type_name='FirebasedataconnectProjectsLocationsServicesSchemasDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Schema.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesSchemasGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Schema) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/schemas/{schemasId}', http_method='GET', method_id='firebasedataconnect.projects.locations.services.schemas.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='FirebasedataconnectProjectsLocationsServicesSchemasGetRequest', response_type_name='Schema', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Schemas in a given project and location. Note that only `schemas/main` is supported, so this will always return at most one Schema.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesSchemasListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSchemasResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/schemas', http_method='GET', method_id='firebasedataconnect.projects.locations.services.schemas.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/schemas', request_field='', request_type_name='FirebasedataconnectProjectsLocationsServicesSchemasListRequest', response_type_name='ListSchemasResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Schema, and creates a new SchemaRevision with the updated Schema.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesSchemasPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/schemas/{schemasId}', http_method='PATCH', method_id='firebasedataconnect.projects.locations.services.schemas.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'requestId', 'updateMask', 'validateOnly'], relative_path='v1alpha/{+name}', request_field='schema', request_type_name='FirebasedataconnectProjectsLocationsServicesSchemasPatchRequest', response_type_name='Operation', supports_download=False)