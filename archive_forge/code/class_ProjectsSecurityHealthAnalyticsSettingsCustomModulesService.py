from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1 import securitycenter_v1_messages as messages
class ProjectsSecurityHealthAnalyticsSettingsCustomModulesService(base_api.BaseApiService):
    """Service class for the projects_securityHealthAnalyticsSettings_customModules resource."""
    _NAME = 'projects_securityHealthAnalyticsSettings_customModules'

    def __init__(self, client):
        super(SecuritycenterV1.ProjectsSecurityHealthAnalyticsSettingsCustomModulesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a resident SecurityHealthAnalyticsCustomModule at the scope of the given CRM parent, and also creates inherited SecurityHealthAnalyticsCustomModules for all CRM descendants of the given parent. These modules are enabled by default.

      Args:
        request: (SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/securityHealthAnalyticsSettings/customModules', http_method='POST', method_id='securitycenter.projects.securityHealthAnalyticsSettings.customModules.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/customModules', request_field='googleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule', request_type_name='SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesCreateRequest', response_type_name='GoogleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified SecurityHealthAnalyticsCustomModule and all of its descendants in the CRM hierarchy. This method is only supported for resident custom modules.

      Args:
        request: (SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/securityHealthAnalyticsSettings/customModules/{customModulesId}', http_method='DELETE', method_id='securitycenter.projects.securityHealthAnalyticsSettings.customModules.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a SecurityHealthAnalyticsCustomModule.

      Args:
        request: (SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/securityHealthAnalyticsSettings/customModules/{customModulesId}', http_method='GET', method_id='securitycenter.projects.securityHealthAnalyticsSettings.customModules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesGetRequest', response_type_name='GoogleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule', supports_download=False)

    def List(self, request, global_params=None):
        """Returns a list of all SecurityHealthAnalyticsCustomModules for the given parent. This includes resident modules defined at the scope of the parent, and inherited modules, inherited from CRM ancestors.

      Args:
        request: (SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSecurityHealthAnalyticsCustomModulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/securityHealthAnalyticsSettings/customModules', http_method='GET', method_id='securitycenter.projects.securityHealthAnalyticsSettings.customModules.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/customModules', request_field='', request_type_name='SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesListRequest', response_type_name='ListSecurityHealthAnalyticsCustomModulesResponse', supports_download=False)

    def ListDescendant(self, request, global_params=None):
        """Returns a list of all resident SecurityHealthAnalyticsCustomModules under the given CRM parent and all of the parent's CRM descendants.

      Args:
        request: (SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesListDescendantRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDescendantSecurityHealthAnalyticsCustomModulesResponse) The response message.
      """
        config = self.GetMethodConfig('ListDescendant')
        return self._RunMethod(config, request, global_params=global_params)
    ListDescendant.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/securityHealthAnalyticsSettings/customModules:listDescendant', http_method='GET', method_id='securitycenter.projects.securityHealthAnalyticsSettings.customModules.listDescendant', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/customModules:listDescendant', request_field='', request_type_name='SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesListDescendantRequest', response_type_name='ListDescendantSecurityHealthAnalyticsCustomModulesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the SecurityHealthAnalyticsCustomModule under the given name based on the given update mask. Updating the enablement state is supported on both resident and inherited modules (though resident modules cannot have an enablement state of "inherited"). Updating the display name and custom config of a module is supported on resident modules only.

      Args:
        request: (SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/securityHealthAnalyticsSettings/customModules/{customModulesId}', http_method='PATCH', method_id='securitycenter.projects.securityHealthAnalyticsSettings.customModules.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule', request_type_name='SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesPatchRequest', response_type_name='GoogleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule', supports_download=False)

    def Simulate(self, request, global_params=None):
        """Simulates a given SecurityHealthAnalyticsCustomModule and Resource.

      Args:
        request: (SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesSimulateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SimulateSecurityHealthAnalyticsCustomModuleResponse) The response message.
      """
        config = self.GetMethodConfig('Simulate')
        return self._RunMethod(config, request, global_params=global_params)
    Simulate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/securityHealthAnalyticsSettings/customModules:simulate', http_method='POST', method_id='securitycenter.projects.securityHealthAnalyticsSettings.customModules.simulate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/customModules:simulate', request_field='simulateSecurityHealthAnalyticsCustomModuleRequest', request_type_name='SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesSimulateRequest', response_type_name='SimulateSecurityHealthAnalyticsCustomModuleResponse', supports_download=False)

    def Test(self, request, global_params=None):
        """Tests a specified or given SecurityHealthAnalyticsCustomModule.

      Args:
        request: (SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesTestRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestSecurityHealthAnalyticsCustomModuleResponse) The response message.
      """
        config = self.GetMethodConfig('Test')
        return self._RunMethod(config, request, global_params=global_params)
    Test.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/securityHealthAnalyticsSettings/customModules/{customModulesId}:test', http_method='POST', method_id='securitycenter.projects.securityHealthAnalyticsSettings.customModules.test', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:test', request_field='testSecurityHealthAnalyticsCustomModuleRequest', request_type_name='SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesTestRequest', response_type_name='TestSecurityHealthAnalyticsCustomModuleResponse', supports_download=False)