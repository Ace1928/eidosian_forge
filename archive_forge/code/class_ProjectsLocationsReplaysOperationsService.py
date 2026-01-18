from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.policysimulator.v1beta import policysimulator_v1beta_messages as messages
class ProjectsLocationsReplaysOperationsService(base_api.BaseApiService):
    """Service class for the projects_locations_replays_operations resource."""
    _NAME = 'projects_locations_replays_operations'

    def __init__(self, client):
        super(PolicysimulatorV1beta.ProjectsLocationsReplaysOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (PolicysimulatorProjectsLocationsReplaysOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/replays/{replaysId}/operations/{operationsId}', http_method='GET', method_id='policysimulator.projects.locations.replays.operations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='PolicysimulatorProjectsLocationsReplaysOperationsGetRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists operations that match the specified filter in the request. If the server doesn't support this method, it returns `UNIMPLEMENTED`.

      Args:
        request: (PolicysimulatorProjectsLocationsReplaysOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningListOperationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/replays/{replaysId}/operations', http_method='GET', method_id='policysimulator.projects.locations.replays.operations.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1beta/{+name}', request_field='', request_type_name='PolicysimulatorProjectsLocationsReplaysOperationsListRequest', response_type_name='GoogleLongrunningListOperationsResponse', supports_download=False)