from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkconnectivity.v1 import networkconnectivity_v1_messages as messages
class ProjectsLocationsInternalRangesService(base_api.BaseApiService):
    """Service class for the projects_locations_internalRanges resource."""
    _NAME = 'projects_locations_internalRanges'

    def __init__(self, client):
        super(NetworkconnectivityV1.ProjectsLocationsInternalRangesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new internal range in a given project and location.

      Args:
        request: (NetworkconnectivityProjectsLocationsInternalRangesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/internalRanges', http_method='POST', method_id='networkconnectivity.projects.locations.internalRanges.create', ordered_params=['parent'], path_params=['parent'], query_params=['internalRangeId', 'requestId'], relative_path='v1/{+parent}/internalRanges', request_field='internalRange', request_type_name='NetworkconnectivityProjectsLocationsInternalRangesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single internal range.

      Args:
        request: (NetworkconnectivityProjectsLocationsInternalRangesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/internalRanges/{internalRangesId}', http_method='DELETE', method_id='networkconnectivity.projects.locations.internalRanges.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsInternalRangesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single internal range.

      Args:
        request: (NetworkconnectivityProjectsLocationsInternalRangesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InternalRange) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/internalRanges/{internalRangesId}', http_method='GET', method_id='networkconnectivity.projects.locations.internalRanges.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsInternalRangesGetRequest', response_type_name='InternalRange', supports_download=False)

    def List(self, request, global_params=None):
        """Lists internal ranges in a given project and location.

      Args:
        request: (NetworkconnectivityProjectsLocationsInternalRangesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListInternalRangesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/internalRanges', http_method='GET', method_id='networkconnectivity.projects.locations.internalRanges.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/internalRanges', request_field='', request_type_name='NetworkconnectivityProjectsLocationsInternalRangesListRequest', response_type_name='ListInternalRangesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single internal range.

      Args:
        request: (NetworkconnectivityProjectsLocationsInternalRangesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/internalRanges/{internalRangesId}', http_method='PATCH', method_id='networkconnectivity.projects.locations.internalRanges.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='internalRange', request_type_name='NetworkconnectivityProjectsLocationsInternalRangesPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)