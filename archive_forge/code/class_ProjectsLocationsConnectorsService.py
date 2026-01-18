from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vpcaccess.v1 import vpcaccess_v1_messages as messages
class ProjectsLocationsConnectorsService(base_api.BaseApiService):
    """Service class for the projects_locations_connectors resource."""
    _NAME = 'projects_locations_connectors'

    def __init__(self, client):
        super(VpcaccessV1.ProjectsLocationsConnectorsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Serverless VPC Access connector, returns an operation.

      Args:
        request: (VpcaccessProjectsLocationsConnectorsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/connectors', http_method='POST', method_id='vpcaccess.projects.locations.connectors.create', ordered_params=['parent'], path_params=['parent'], query_params=['connectorId'], relative_path='v1/{+parent}/connectors', request_field='connector', request_type_name='VpcaccessProjectsLocationsConnectorsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Serverless VPC Access connector. Returns NOT_FOUND if the resource does not exist.

      Args:
        request: (VpcaccessProjectsLocationsConnectorsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/connectors/{connectorsId}', http_method='DELETE', method_id='vpcaccess.projects.locations.connectors.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VpcaccessProjectsLocationsConnectorsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a Serverless VPC Access connector. Returns NOT_FOUND if the resource does not exist.

      Args:
        request: (VpcaccessProjectsLocationsConnectorsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Connector) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/connectors/{connectorsId}', http_method='GET', method_id='vpcaccess.projects.locations.connectors.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VpcaccessProjectsLocationsConnectorsGetRequest', response_type_name='Connector', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Serverless VPC Access connectors.

      Args:
        request: (VpcaccessProjectsLocationsConnectorsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConnectorsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/connectors', http_method='GET', method_id='vpcaccess.projects.locations.connectors.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/connectors', request_field='', request_type_name='VpcaccessProjectsLocationsConnectorsListRequest', response_type_name='ListConnectorsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a Serverless VPC Access connector, returns an operation.

      Args:
        request: (VpcaccessProjectsLocationsConnectorsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/connectors/{connectorsId}', http_method='PATCH', method_id='vpcaccess.projects.locations.connectors.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='connector', request_type_name='VpcaccessProjectsLocationsConnectorsPatchRequest', response_type_name='Operation', supports_download=False)