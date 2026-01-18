from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v3 import monitoring_v3_messages as messages
class ProjectsAlertPoliciesService(base_api.BaseApiService):
    """Service class for the projects_alertPolicies resource."""
    _NAME = 'projects_alertPolicies'

    def __init__(self, client):
        super(MonitoringV3.ProjectsAlertPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new alerting policy.Design your application to single-thread API calls that modify the state of alerting policies in a single project. This includes calls to CreateAlertPolicy, DeleteAlertPolicy and UpdateAlertPolicy.

      Args:
        request: (MonitoringProjectsAlertPoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AlertPolicy) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/alertPolicies', http_method='POST', method_id='monitoring.projects.alertPolicies.create', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}/alertPolicies', request_field='alertPolicy', request_type_name='MonitoringProjectsAlertPoliciesCreateRequest', response_type_name='AlertPolicy', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an alerting policy.Design your application to single-thread API calls that modify the state of alerting policies in a single project. This includes calls to CreateAlertPolicy, DeleteAlertPolicy and UpdateAlertPolicy.

      Args:
        request: (MonitoringProjectsAlertPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/alertPolicies/{alertPoliciesId}', http_method='DELETE', method_id='monitoring.projects.alertPolicies.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='MonitoringProjectsAlertPoliciesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a single alerting policy.

      Args:
        request: (MonitoringProjectsAlertPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AlertPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/alertPolicies/{alertPoliciesId}', http_method='GET', method_id='monitoring.projects.alertPolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='MonitoringProjectsAlertPoliciesGetRequest', response_type_name='AlertPolicy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the existing alerting policies for the workspace.

      Args:
        request: (MonitoringProjectsAlertPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAlertPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/alertPolicies', http_method='GET', method_id='monitoring.projects.alertPolicies.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v3/{+name}/alertPolicies', request_field='', request_type_name='MonitoringProjectsAlertPoliciesListRequest', response_type_name='ListAlertPoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an alerting policy. You can either replace the entire policy with a new one or replace only certain fields in the current alerting policy by specifying the fields to be updated via updateMask. Returns the updated alerting policy.Design your application to single-thread API calls that modify the state of alerting policies in a single project. This includes calls to CreateAlertPolicy, DeleteAlertPolicy and UpdateAlertPolicy.

      Args:
        request: (MonitoringProjectsAlertPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AlertPolicy) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/alertPolicies/{alertPoliciesId}', http_method='PATCH', method_id='monitoring.projects.alertPolicies.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v3/{+name}', request_field='alertPolicy', request_type_name='MonitoringProjectsAlertPoliciesPatchRequest', response_type_name='AlertPolicy', supports_download=False)