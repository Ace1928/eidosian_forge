from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouddeploy.v1 import clouddeploy_v1_messages as messages
class ProjectsLocationsDeliveryPipelinesAutomationRunsService(base_api.BaseApiService):
    """Service class for the projects_locations_deliveryPipelines_automationRuns resource."""
    _NAME = 'projects_locations_deliveryPipelines_automationRuns'

    def __init__(self, client):
        super(ClouddeployV1.ProjectsLocationsDeliveryPipelinesAutomationRunsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancels an AutomationRun. The `state` of the `AutomationRun` after cancelling is `CANCELLED`. `CancelAutomationRun` can be called on AutomationRun in the state `IN_PROGRESS` and `PENDING`; AutomationRun in a different state returns an `FAILED_PRECONDITION` error.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesAutomationRunsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CancelAutomationRunResponse) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/automationRuns/{automationRunsId}:cancel', http_method='POST', method_id='clouddeploy.projects.locations.deliveryPipelines.automationRuns.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='cancelAutomationRunRequest', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesAutomationRunsCancelRequest', response_type_name='CancelAutomationRunResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single AutomationRun.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesAutomationRunsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AutomationRun) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/automationRuns/{automationRunsId}', http_method='GET', method_id='clouddeploy.projects.locations.deliveryPipelines.automationRuns.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesAutomationRunsGetRequest', response_type_name='AutomationRun', supports_download=False)

    def List(self, request, global_params=None):
        """Lists AutomationRuns in a given project and location.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesAutomationRunsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAutomationRunsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/automationRuns', http_method='GET', method_id='clouddeploy.projects.locations.deliveryPipelines.automationRuns.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/automationRuns', request_field='', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesAutomationRunsListRequest', response_type_name='ListAutomationRunsResponse', supports_download=False)