from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.marketplacesolutions.v1alpha1 import marketplacesolutions_v1alpha1_messages as messages
class ProjectsLocationsPowerInstancesService(base_api.BaseApiService):
    """Service class for the projects_locations_powerInstances resource."""
    _NAME = 'projects_locations_powerInstances'

    def __init__(self, client):
        super(MarketplacesolutionsV1alpha1.ProjectsLocationsPowerInstancesService, self).__init__(client)
        self._upload_configs = {}

    def ApplyPowerAction(self, request, global_params=None):
        """Performs one of several power-related actions on an instance.

      Args:
        request: (MarketplacesolutionsProjectsLocationsPowerInstancesApplyPowerActionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ApplyPowerAction')
        return self._RunMethod(config, request, global_params=global_params)
    ApplyPowerAction.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/powerInstances/{powerInstancesId}:applyPowerAction', http_method='POST', method_id='marketplacesolutions.projects.locations.powerInstances.applyPowerAction', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:applyPowerAction', request_field='applyPowerInstancePowerActionRequest', request_type_name='MarketplacesolutionsProjectsLocationsPowerInstancesApplyPowerActionRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Create a Power instance.

      Args:
        request: (MarketplacesolutionsProjectsLocationsPowerInstancesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/powerInstances', http_method='POST', method_id='marketplacesolutions.projects.locations.powerInstances.create', ordered_params=['parent'], path_params=['parent'], query_params=['powerInstanceId'], relative_path='v1alpha1/{+parent}/powerInstances', request_field='powerInstance', request_type_name='MarketplacesolutionsProjectsLocationsPowerInstancesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete a Power instance.

      Args:
        request: (MarketplacesolutionsProjectsLocationsPowerInstancesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/powerInstances/{powerInstancesId}', http_method='DELETE', method_id='marketplacesolutions.projects.locations.powerInstances.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='MarketplacesolutionsProjectsLocationsPowerInstancesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Get details about a single server from Power.

      Args:
        request: (MarketplacesolutionsProjectsLocationsPowerInstancesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PowerInstance) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/powerInstances/{powerInstancesId}', http_method='GET', method_id='marketplacesolutions.projects.locations.powerInstances.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='MarketplacesolutionsProjectsLocationsPowerInstancesGetRequest', response_type_name='PowerInstance', supports_download=False)

    def List(self, request, global_params=None):
        """List servers in a given project and location from Power.

      Args:
        request: (MarketplacesolutionsProjectsLocationsPowerInstancesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPowerInstancesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/powerInstances', http_method='GET', method_id='marketplacesolutions.projects.locations.powerInstances.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/powerInstances', request_field='', request_type_name='MarketplacesolutionsProjectsLocationsPowerInstancesListRequest', response_type_name='ListPowerInstancesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update a Power instance.

      Args:
        request: (MarketplacesolutionsProjectsLocationsPowerInstancesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/powerInstances/{powerInstancesId}', http_method='PATCH', method_id='marketplacesolutions.projects.locations.powerInstances.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='powerInstance', request_type_name='MarketplacesolutionsProjectsLocationsPowerInstancesPatchRequest', response_type_name='Operation', supports_download=False)

    def Reset(self, request, global_params=None):
        """Reset a running instance's state.

      Args:
        request: (MarketplacesolutionsProjectsLocationsPowerInstancesResetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Reset')
        return self._RunMethod(config, request, global_params=global_params)
    Reset.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/powerInstances/{powerInstancesId}:reset', http_method='POST', method_id='marketplacesolutions.projects.locations.powerInstances.reset', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:reset', request_field='resetPowerInstanceRequest', request_type_name='MarketplacesolutionsProjectsLocationsPowerInstancesResetRequest', response_type_name='Operation', supports_download=False)