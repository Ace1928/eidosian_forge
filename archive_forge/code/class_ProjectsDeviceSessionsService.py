from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.testing.v1 import testing_v1_messages as messages
class ProjectsDeviceSessionsService(base_api.BaseApiService):
    """Service class for the projects_deviceSessions resource."""
    _NAME = 'projects_deviceSessions'

    def __init__(self, client):
        super(TestingV1.ProjectsDeviceSessionsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """POST /v1/projects/{project_id}/deviceSessions/{device_session_id}:cancel Changes the DeviceSession to state FINISHED and terminates all connections. Canceled sessions are not deleted and can be retrieved or listed by the user until they expire based on the 28 day deletion policy.

      Args:
        request: (TestingProjectsDeviceSessionsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/deviceSessions/{deviceSessionsId}:cancel', http_method='POST', method_id='testing.projects.deviceSessions.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='cancelDeviceSessionRequest', request_type_name='TestingProjectsDeviceSessionsCancelRequest', response_type_name='Empty', supports_download=False)

    def Create(self, request, global_params=None):
        """POST /v1/projects/{project_id}/deviceSessions.

      Args:
        request: (TestingProjectsDeviceSessionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DeviceSession) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/deviceSessions', http_method='POST', method_id='testing.projects.deviceSessions.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/deviceSessions', request_field='deviceSession', request_type_name='TestingProjectsDeviceSessionsCreateRequest', response_type_name='DeviceSession', supports_download=False)

    def Get(self, request, global_params=None):
        """GET /v1/projects/{project_id}/deviceSessions/{device_session_id} Return a DeviceSession, which documents the allocation status and whether the device is allocated. Clients making requests from this API must poll GetDeviceSession.

      Args:
        request: (TestingProjectsDeviceSessionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DeviceSession) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/deviceSessions/{deviceSessionsId}', http_method='GET', method_id='testing.projects.deviceSessions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='TestingProjectsDeviceSessionsGetRequest', response_type_name='DeviceSession', supports_download=False)

    def List(self, request, global_params=None):
        """GET /v1/projects/{project_id}/deviceSessions Lists device Sessions owned by the project user.

      Args:
        request: (TestingProjectsDeviceSessionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDeviceSessionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/deviceSessions', http_method='GET', method_id='testing.projects.deviceSessions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/deviceSessions', request_field='', request_type_name='TestingProjectsDeviceSessionsListRequest', response_type_name='ListDeviceSessionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """PATCH /v1/projects/{projectId}/deviceSessions/deviceSessionId}:updateDeviceSession Updates the current device session to the fields described by the update_mask.

      Args:
        request: (TestingProjectsDeviceSessionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DeviceSession) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/deviceSessions/{deviceSessionsId}', http_method='PATCH', method_id='testing.projects.deviceSessions.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='deviceSession', request_type_name='TestingProjectsDeviceSessionsPatchRequest', response_type_name='DeviceSession', supports_download=False)