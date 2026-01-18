from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v2alpha import apigee_v2alpha_messages as messages
class ProjectsLocationsDummiesService(base_api.BaseApiService):
    """Service class for the projects_locations_dummies resource."""
    _NAME = 'projects_locations_dummies'

    def __init__(self, client):
        super(ApigeeV2alpha.ProjectsLocationsDummiesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Dummy in a given project and location.

      Args:
        request: (ApigeeProjectsLocationsDummiesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2alpha/projects/{projectsId}/locations/{locationsId}/dummies', http_method='POST', method_id='apigee.projects.locations.dummies.create', ordered_params=['parent'], path_params=['parent'], query_params=['dummyId', 'requestId'], relative_path='v2alpha/{+parent}/dummies', request_field='dummy', request_type_name='ApigeeProjectsLocationsDummiesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Dummy.

      Args:
        request: (ApigeeProjectsLocationsDummiesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2alpha/projects/{projectsId}/locations/{locationsId}/dummies/{dummiesId}', http_method='DELETE', method_id='apigee.projects.locations.dummies.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v2alpha/{+name}', request_field='', request_type_name='ApigeeProjectsLocationsDummiesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Dummy.

      Args:
        request: (ApigeeProjectsLocationsDummiesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Dummy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2alpha/projects/{projectsId}/locations/{locationsId}/dummies/{dummiesId}', http_method='GET', method_id='apigee.projects.locations.dummies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2alpha/{+name}', request_field='', request_type_name='ApigeeProjectsLocationsDummiesGetRequest', response_type_name='Dummy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Dummies in a given project and location.

      Args:
        request: (ApigeeProjectsLocationsDummiesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDummiesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2alpha/projects/{projectsId}/locations/{locationsId}/dummies', http_method='GET', method_id='apigee.projects.locations.dummies.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v2alpha/{+parent}/dummies', request_field='', request_type_name='ApigeeProjectsLocationsDummiesListRequest', response_type_name='ListDummiesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Dummy.

      Args:
        request: (ApigeeProjectsLocationsDummiesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2alpha/projects/{projectsId}/locations/{locationsId}/dummies/{dummiesId}', http_method='PATCH', method_id='apigee.projects.locations.dummies.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v2alpha/{+name}', request_field='dummy', request_type_name='ApigeeProjectsLocationsDummiesPatchRequest', response_type_name='Operation', supports_download=False)