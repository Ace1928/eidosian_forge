from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsApisRevisionsDebugsessionsService(base_api.BaseApiService):
    """Service class for the organizations_environments_apis_revisions_debugsessions resource."""
    _NAME = 'organizations_environments_apis_revisions_debugsessions'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsApisRevisionsDebugsessionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a debug session for a deployed API Proxy revision.

      Args:
        request: (ApigeeOrganizationsEnvironmentsApisRevisionsDebugsessionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DebugSession) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/apis/{apisId}/revisions/{revisionsId}/debugsessions', http_method='POST', method_id='apigee.organizations.environments.apis.revisions.debugsessions.create', ordered_params=['parent'], path_params=['parent'], query_params=['timeout'], relative_path='v1/{+parent}/debugsessions', request_field='googleCloudApigeeV1DebugSession', request_type_name='ApigeeOrganizationsEnvironmentsApisRevisionsDebugsessionsCreateRequest', response_type_name='GoogleCloudApigeeV1DebugSession', supports_download=False)

    def DeleteData(self, request, global_params=None):
        """Deletes the data from a debug session. This does not cancel the debug session or prevent further data from being collected if the session is still active in runtime pods.

      Args:
        request: (ApigeeOrganizationsEnvironmentsApisRevisionsDebugsessionsDeleteDataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('DeleteData')
        return self._RunMethod(config, request, global_params=global_params)
    DeleteData.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/apis/{apisId}/revisions/{revisionsId}/debugsessions/{debugsessionsId}/data', http_method='DELETE', method_id='apigee.organizations.environments.apis.revisions.debugsessions.deleteData', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}/data', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsApisRevisionsDebugsessionsDeleteDataRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a debug session.

      Args:
        request: (ApigeeOrganizationsEnvironmentsApisRevisionsDebugsessionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DebugSession) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/apis/{apisId}/revisions/{revisionsId}/debugsessions/{debugsessionsId}', http_method='GET', method_id='apigee.organizations.environments.apis.revisions.debugsessions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsApisRevisionsDebugsessionsGetRequest', response_type_name='GoogleCloudApigeeV1DebugSession', supports_download=False)

    def List(self, request, global_params=None):
        """Lists debug sessions that are currently active in the given API Proxy revision.

      Args:
        request: (ApigeeOrganizationsEnvironmentsApisRevisionsDebugsessionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListDebugSessionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/apis/{apisId}/revisions/{revisionsId}/debugsessions', http_method='GET', method_id='apigee.organizations.environments.apis.revisions.debugsessions.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/debugsessions', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsApisRevisionsDebugsessionsListRequest', response_type_name='GoogleCloudApigeeV1ListDebugSessionsResponse', supports_download=False)