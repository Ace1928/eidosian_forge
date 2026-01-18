from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1 import networkservices_v1_messages as messages
class ProjectsLocationsMulticastConsumerAssociationsService(base_api.BaseApiService):
    """Service class for the projects_locations_multicastConsumerAssociations resource."""
    _NAME = 'projects_locations_multicastConsumerAssociations'

    def __init__(self, client):
        super(NetworkservicesV1.ProjectsLocationsMulticastConsumerAssociationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new multicast consumer association in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastConsumerAssociationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastConsumerAssociations', http_method='POST', method_id='networkservices.projects.locations.multicastConsumerAssociations.create', ordered_params=['parent'], path_params=['parent'], query_params=['multicastConsumerAssociationId', 'requestId'], relative_path='v1/{+parent}/multicastConsumerAssociations', request_field='multicastConsumerAssociation', request_type_name='NetworkservicesProjectsLocationsMulticastConsumerAssociationsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single multicast consumer association.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastConsumerAssociationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastConsumerAssociations/{multicastConsumerAssociationsId}', http_method='DELETE', method_id='networkservices.projects.locations.multicastConsumerAssociations.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastConsumerAssociationsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single multicast consumer association.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastConsumerAssociationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MulticastConsumerAssociation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastConsumerAssociations/{multicastConsumerAssociationsId}', http_method='GET', method_id='networkservices.projects.locations.multicastConsumerAssociations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastConsumerAssociationsGetRequest', response_type_name='MulticastConsumerAssociation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists multicast consumer associations in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastConsumerAssociationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMulticastConsumerAssociationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastConsumerAssociations', http_method='GET', method_id='networkservices.projects.locations.multicastConsumerAssociations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/multicastConsumerAssociations', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastConsumerAssociationsListRequest', response_type_name='ListMulticastConsumerAssociationsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single multicast consumer association.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastConsumerAssociationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastConsumerAssociations/{multicastConsumerAssociationsId}', http_method='PATCH', method_id='networkservices.projects.locations.multicastConsumerAssociations.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='multicastConsumerAssociation', request_type_name='NetworkservicesProjectsLocationsMulticastConsumerAssociationsPatchRequest', response_type_name='Operation', supports_download=False)