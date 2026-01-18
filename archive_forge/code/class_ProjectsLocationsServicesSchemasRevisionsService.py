from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firebasedataconnect.v1alpha import firebasedataconnect_v1alpha_messages as messages
class ProjectsLocationsServicesSchemasRevisionsService(base_api.BaseApiService):
    """Service class for the projects_locations_services_schemas_revisions resource."""
    _NAME = 'projects_locations_services_schemas_revisions'

    def __init__(self, client):
        super(FirebasedataconnectV1alpha.ProjectsLocationsServicesSchemasRevisionsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a single SchemaRevision.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesSchemasRevisionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/schemas/{schemasId}/revisions/{revisionsId}', http_method='DELETE', method_id='firebasedataconnect.projects.locations.services.schemas.revisions.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'requestId', 'validateOnly'], relative_path='v1alpha/{+name}', request_field='', request_type_name='FirebasedataconnectProjectsLocationsServicesSchemasRevisionsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single SchemaRevision.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesSchemasRevisionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SchemaRevision) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/schemas/{schemasId}/revisions/{revisionsId}', http_method='GET', method_id='firebasedataconnect.projects.locations.services.schemas.revisions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='FirebasedataconnectProjectsLocationsServicesSchemasRevisionsGetRequest', response_type_name='SchemaRevision', supports_download=False)

    def List(self, request, global_params=None):
        """Lists SchemaRevisions in a given project and location.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesSchemasRevisionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSchemaRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/schemas/{schemasId}/revisions', http_method='GET', method_id='firebasedataconnect.projects.locations.services.schemas.revisions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/revisions', request_field='', request_type_name='FirebasedataconnectProjectsLocationsServicesSchemasRevisionsListRequest', response_type_name='ListSchemaRevisionsResponse', supports_download=False)