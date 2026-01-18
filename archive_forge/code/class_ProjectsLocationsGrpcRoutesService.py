from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1 import networkservices_v1_messages as messages
class ProjectsLocationsGrpcRoutesService(base_api.BaseApiService):
    """Service class for the projects_locations_grpcRoutes resource."""
    _NAME = 'projects_locations_grpcRoutes'

    def __init__(self, client):
        super(NetworkservicesV1.ProjectsLocationsGrpcRoutesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new GrpcRoute in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsGrpcRoutesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/grpcRoutes', http_method='POST', method_id='networkservices.projects.locations.grpcRoutes.create', ordered_params=['parent'], path_params=['parent'], query_params=['grpcRouteId'], relative_path='v1/{+parent}/grpcRoutes', request_field='grpcRoute', request_type_name='NetworkservicesProjectsLocationsGrpcRoutesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single GrpcRoute.

      Args:
        request: (NetworkservicesProjectsLocationsGrpcRoutesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/grpcRoutes/{grpcRoutesId}', http_method='DELETE', method_id='networkservices.projects.locations.grpcRoutes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsGrpcRoutesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single GrpcRoute.

      Args:
        request: (NetworkservicesProjectsLocationsGrpcRoutesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GrpcRoute) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/grpcRoutes/{grpcRoutesId}', http_method='GET', method_id='networkservices.projects.locations.grpcRoutes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsGrpcRoutesGetRequest', response_type_name='GrpcRoute', supports_download=False)

    def List(self, request, global_params=None):
        """Lists GrpcRoutes in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsGrpcRoutesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGrpcRoutesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/grpcRoutes', http_method='GET', method_id='networkservices.projects.locations.grpcRoutes.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/grpcRoutes', request_field='', request_type_name='NetworkservicesProjectsLocationsGrpcRoutesListRequest', response_type_name='ListGrpcRoutesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single GrpcRoute.

      Args:
        request: (NetworkservicesProjectsLocationsGrpcRoutesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/grpcRoutes/{grpcRoutesId}', http_method='PATCH', method_id='networkservices.projects.locations.grpcRoutes.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='grpcRoute', request_type_name='NetworkservicesProjectsLocationsGrpcRoutesPatchRequest', response_type_name='Operation', supports_download=False)