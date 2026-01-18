from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycentermanagement.v1 import securitycentermanagement_v1_messages as messages
class OrganizationsLocationsSecurityHealthAnalyticsCustomModulesService(base_api.BaseApiService):
    """Service class for the organizations_locations_securityHealthAnalyticsCustomModules resource."""
    _NAME = 'organizations_locations_securityHealthAnalyticsCustomModules'

    def __init__(self, client):
        super(SecuritycentermanagementV1.OrganizationsLocationsSecurityHealthAnalyticsCustomModulesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a resident SecurityHealthAnalyticsCustomModule at the scope of the given CRM parent, and also creates inherited SecurityHealthAnalyticsCustomModules for all CRM descendants of the given parent. These modules are enabled by default.

      Args:
        request: (SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsCustomModulesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityHealthAnalyticsCustomModule) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/securityHealthAnalyticsCustomModules', http_method='POST', method_id='securitycentermanagement.organizations.locations.securityHealthAnalyticsCustomModules.create', ordered_params=['parent'], path_params=['parent'], query_params=['validateOnly'], relative_path='v1/{+parent}/securityHealthAnalyticsCustomModules', request_field='securityHealthAnalyticsCustomModule', request_type_name='SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsCustomModulesCreateRequest', response_type_name='SecurityHealthAnalyticsCustomModule', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified SecurityHealthAnalyticsCustomModule and all of its descendants in the CRM hierarchy. This method is only supported for resident custom modules.

      Args:
        request: (SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsCustomModulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/securityHealthAnalyticsCustomModules/{securityHealthAnalyticsCustomModulesId}', http_method='DELETE', method_id='securitycentermanagement.organizations.locations.securityHealthAnalyticsCustomModules.delete', ordered_params=['name'], path_params=['name'], query_params=['validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsCustomModulesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a SecurityHealthAnalyticsCustomModule.

      Args:
        request: (SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsCustomModulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityHealthAnalyticsCustomModule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/securityHealthAnalyticsCustomModules/{securityHealthAnalyticsCustomModulesId}', http_method='GET', method_id='securitycentermanagement.organizations.locations.securityHealthAnalyticsCustomModules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsCustomModulesGetRequest', response_type_name='SecurityHealthAnalyticsCustomModule', supports_download=False)

    def List(self, request, global_params=None):
        """Returns a list of all SecurityHealthAnalyticsCustomModules for the given parent. This includes resident modules defined at the scope of the parent, and inherited modules, inherited from CRM ancestors (no descendants).

      Args:
        request: (SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsCustomModulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSecurityHealthAnalyticsCustomModulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/securityHealthAnalyticsCustomModules', http_method='GET', method_id='securitycentermanagement.organizations.locations.securityHealthAnalyticsCustomModules.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/securityHealthAnalyticsCustomModules', request_field='', request_type_name='SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsCustomModulesListRequest', response_type_name='ListSecurityHealthAnalyticsCustomModulesResponse', supports_download=False)

    def ListDescendant(self, request, global_params=None):
        """Returns a list of all resident SecurityHealthAnalyticsCustomModules under the given CRM parent and all of the parent's CRM descendants.

      Args:
        request: (SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsCustomModulesListDescendantRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDescendantSecurityHealthAnalyticsCustomModulesResponse) The response message.
      """
        config = self.GetMethodConfig('ListDescendant')
        return self._RunMethod(config, request, global_params=global_params)
    ListDescendant.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/securityHealthAnalyticsCustomModules:listDescendant', http_method='GET', method_id='securitycentermanagement.organizations.locations.securityHealthAnalyticsCustomModules.listDescendant', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/securityHealthAnalyticsCustomModules:listDescendant', request_field='', request_type_name='SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsCustomModulesListDescendantRequest', response_type_name='ListDescendantSecurityHealthAnalyticsCustomModulesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the SecurityHealthAnalyticsCustomModule under the given name based on the given update mask. Updating the enablement state is supported on both resident and inherited modules (though resident modules cannot have an enablement state of "inherited"). Updating the display name and custom config of a module is supported on resident modules only.

      Args:
        request: (SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsCustomModulesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityHealthAnalyticsCustomModule) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/securityHealthAnalyticsCustomModules/{securityHealthAnalyticsCustomModulesId}', http_method='PATCH', method_id='securitycentermanagement.organizations.locations.securityHealthAnalyticsCustomModules.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='securityHealthAnalyticsCustomModule', request_type_name='SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsCustomModulesPatchRequest', response_type_name='SecurityHealthAnalyticsCustomModule', supports_download=False)

    def Simulate(self, request, global_params=None):
        """Simulates a given SecurityHealthAnalyticsCustomModule and Resource.

      Args:
        request: (SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsCustomModulesSimulateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SimulateSecurityHealthAnalyticsCustomModuleResponse) The response message.
      """
        config = self.GetMethodConfig('Simulate')
        return self._RunMethod(config, request, global_params=global_params)
    Simulate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/securityHealthAnalyticsCustomModules:simulate', http_method='POST', method_id='securitycentermanagement.organizations.locations.securityHealthAnalyticsCustomModules.simulate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/securityHealthAnalyticsCustomModules:simulate', request_field='simulateSecurityHealthAnalyticsCustomModuleRequest', request_type_name='SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsCustomModulesSimulateRequest', response_type_name='SimulateSecurityHealthAnalyticsCustomModuleResponse', supports_download=False)