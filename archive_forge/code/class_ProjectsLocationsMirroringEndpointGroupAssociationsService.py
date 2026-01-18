from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1alpha1 import networksecurity_v1alpha1_messages as messages
class ProjectsLocationsMirroringEndpointGroupAssociationsService(base_api.BaseApiService):
    """Service class for the projects_locations_mirroringEndpointGroupAssociations resource."""
    _NAME = 'projects_locations_mirroringEndpointGroupAssociations'

    def __init__(self, client):
        super(NetworksecurityV1alpha1.ProjectsLocationsMirroringEndpointGroupAssociationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new MirroringEndpointGroupAssociation in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsMirroringEndpointGroupAssociationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/mirroringEndpointGroupAssociations', http_method='POST', method_id='networksecurity.projects.locations.mirroringEndpointGroupAssociations.create', ordered_params=['parent'], path_params=['parent'], query_params=['mirroringEndpointGroupAssociationId', 'requestId'], relative_path='v1alpha1/{+parent}/mirroringEndpointGroupAssociations', request_field='mirroringEndpointGroupAssociation', request_type_name='NetworksecurityProjectsLocationsMirroringEndpointGroupAssociationsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single MirroringEndpointGroupAssociation.

      Args:
        request: (NetworksecurityProjectsLocationsMirroringEndpointGroupAssociationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/mirroringEndpointGroupAssociations/{mirroringEndpointGroupAssociationsId}', http_method='DELETE', method_id='networksecurity.projects.locations.mirroringEndpointGroupAssociations.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsMirroringEndpointGroupAssociationsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single MirroringEndpointGroupAssociation.

      Args:
        request: (NetworksecurityProjectsLocationsMirroringEndpointGroupAssociationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MirroringEndpointGroupAssociation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/mirroringEndpointGroupAssociations/{mirroringEndpointGroupAssociationsId}', http_method='GET', method_id='networksecurity.projects.locations.mirroringEndpointGroupAssociations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsMirroringEndpointGroupAssociationsGetRequest', response_type_name='MirroringEndpointGroupAssociation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists MirroringEndpointGroupAssociations in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsMirroringEndpointGroupAssociationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMirroringEndpointGroupAssociationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/mirroringEndpointGroupAssociations', http_method='GET', method_id='networksecurity.projects.locations.mirroringEndpointGroupAssociations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/mirroringEndpointGroupAssociations', request_field='', request_type_name='NetworksecurityProjectsLocationsMirroringEndpointGroupAssociationsListRequest', response_type_name='ListMirroringEndpointGroupAssociationsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a single MirroringEndpointGroupAssociation.

      Args:
        request: (NetworksecurityProjectsLocationsMirroringEndpointGroupAssociationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/mirroringEndpointGroupAssociations/{mirroringEndpointGroupAssociationsId}', http_method='PATCH', method_id='networksecurity.projects.locations.mirroringEndpointGroupAssociations.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha1/{+name}', request_field='mirroringEndpointGroupAssociation', request_type_name='NetworksecurityProjectsLocationsMirroringEndpointGroupAssociationsPatchRequest', response_type_name='Operation', supports_download=False)