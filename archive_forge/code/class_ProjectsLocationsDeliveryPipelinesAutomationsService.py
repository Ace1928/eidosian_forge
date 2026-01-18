from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouddeploy.v1 import clouddeploy_v1_messages as messages
class ProjectsLocationsDeliveryPipelinesAutomationsService(base_api.BaseApiService):
    """Service class for the projects_locations_deliveryPipelines_automations resource."""
    _NAME = 'projects_locations_deliveryPipelines_automations'

    def __init__(self, client):
        super(ClouddeployV1.ProjectsLocationsDeliveryPipelinesAutomationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Automation in a given project and location.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesAutomationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/automations', http_method='POST', method_id='clouddeploy.projects.locations.deliveryPipelines.automations.create', ordered_params=['parent'], path_params=['parent'], query_params=['automationId', 'requestId', 'validateOnly'], relative_path='v1/{+parent}/automations', request_field='automation', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesAutomationsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Automation resource.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesAutomationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/automations/{automationsId}', http_method='DELETE', method_id='clouddeploy.projects.locations.deliveryPipelines.automations.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'requestId', 'validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesAutomationsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Automation.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesAutomationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Automation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/automations/{automationsId}', http_method='GET', method_id='clouddeploy.projects.locations.deliveryPipelines.automations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesAutomationsGetRequest', response_type_name='Automation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Automations in a given project and location.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesAutomationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAutomationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/automations', http_method='GET', method_id='clouddeploy.projects.locations.deliveryPipelines.automations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/automations', request_field='', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesAutomationsListRequest', response_type_name='ListAutomationsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Automation resource.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesAutomationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deliveryPipelines/{deliveryPipelinesId}/automations/{automationsId}', http_method='PATCH', method_id='clouddeploy.projects.locations.deliveryPipelines.automations.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'requestId', 'updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='automation', request_type_name='ClouddeployProjectsLocationsDeliveryPipelinesAutomationsPatchRequest', response_type_name='Operation', supports_download=False)