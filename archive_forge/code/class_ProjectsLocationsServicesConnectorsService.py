from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firebasedataconnect.v1alpha import firebasedataconnect_v1alpha_messages as messages
class ProjectsLocationsServicesConnectorsService(base_api.BaseApiService):
    """Service class for the projects_locations_services_connectors resource."""
    _NAME = 'projects_locations_services_connectors'

    def __init__(self, client):
        super(FirebasedataconnectV1alpha.ProjectsLocationsServicesConnectorsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Connector in a given project and location. The operations are validated against and must be compatible with the active schema. If the operations and schema are not compatible or if the schema is not present, this will result in an error.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesConnectorsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/connectors', http_method='POST', method_id='firebasedataconnect.projects.locations.services.connectors.create', ordered_params=['parent'], path_params=['parent'], query_params=['connectorId', 'requestId', 'validateOnly'], relative_path='v1alpha/{+parent}/connectors', request_field='connector', request_type_name='FirebasedataconnectProjectsLocationsServicesConnectorsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Connector.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesConnectorsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/connectors/{connectorsId}', http_method='DELETE', method_id='firebasedataconnect.projects.locations.services.connectors.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'force', 'requestId', 'validateOnly'], relative_path='v1alpha/{+name}', request_field='', request_type_name='FirebasedataconnectProjectsLocationsServicesConnectorsDeleteRequest', response_type_name='Operation', supports_download=False)

    def ExecuteMutation(self, request, global_params=None):
        """Execute a predefined mutation in a Connector.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesConnectorsExecuteMutationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExecuteMutationResponse) The response message.
      """
        config = self.GetMethodConfig('ExecuteMutation')
        return self._RunMethod(config, request, global_params=global_params)
    ExecuteMutation.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/connectors/{connectorsId}:executeMutation', http_method='POST', method_id='firebasedataconnect.projects.locations.services.connectors.executeMutation', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}:executeMutation', request_field='executeMutationRequest', request_type_name='FirebasedataconnectProjectsLocationsServicesConnectorsExecuteMutationRequest', response_type_name='ExecuteMutationResponse', supports_download=False)

    def ExecuteQuery(self, request, global_params=None):
        """Execute a predefined query in a Connector.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesConnectorsExecuteQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExecuteQueryResponse) The response message.
      """
        config = self.GetMethodConfig('ExecuteQuery')
        return self._RunMethod(config, request, global_params=global_params)
    ExecuteQuery.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/connectors/{connectorsId}:executeQuery', http_method='POST', method_id='firebasedataconnect.projects.locations.services.connectors.executeQuery', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}:executeQuery', request_field='executeQueryRequest', request_type_name='FirebasedataconnectProjectsLocationsServicesConnectorsExecuteQueryRequest', response_type_name='ExecuteQueryResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Connector.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesConnectorsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Connector) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/connectors/{connectorsId}', http_method='GET', method_id='firebasedataconnect.projects.locations.services.connectors.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='FirebasedataconnectProjectsLocationsServicesConnectorsGetRequest', response_type_name='Connector', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Connectors in a given project and location.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesConnectorsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConnectorsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/connectors', http_method='GET', method_id='firebasedataconnect.projects.locations.services.connectors.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/connectors', request_field='', request_type_name='FirebasedataconnectProjectsLocationsServicesConnectorsListRequest', response_type_name='ListConnectorsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Connector, and creates a new ConnectorRevision with the updated Connector. The operations are validated against and must be compatible with the live schema. If the operations and schema are not compatible or if the schema is not present, this will result in an error.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesConnectorsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/connectors/{connectorsId}', http_method='PATCH', method_id='firebasedataconnect.projects.locations.services.connectors.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'requestId', 'revisionId', 'updateMask', 'validateOnly'], relative_path='v1alpha/{+name}', request_field='connector', request_type_name='FirebasedataconnectProjectsLocationsServicesConnectorsPatchRequest', response_type_name='Operation', supports_download=False)