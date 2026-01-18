from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1 import securitycenter_v1_messages as messages
class ProjectsSecurityHealthAnalyticsSettingsEffectiveCustomModulesService(base_api.BaseApiService):
    """Service class for the projects_securityHealthAnalyticsSettings_effectiveCustomModules resource."""
    _NAME = 'projects_securityHealthAnalyticsSettings_effectiveCustomModules'

    def __init__(self, client):
        super(SecuritycenterV1.ProjectsSecurityHealthAnalyticsSettingsEffectiveCustomModulesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves an EffectiveSecurityHealthAnalyticsCustomModule.

      Args:
        request: (SecuritycenterProjectsSecurityHealthAnalyticsSettingsEffectiveCustomModulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV1EffectiveSecurityHealthAnalyticsCustomModule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/securityHealthAnalyticsSettings/effectiveCustomModules/{effectiveCustomModulesId}', http_method='GET', method_id='securitycenter.projects.securityHealthAnalyticsSettings.effectiveCustomModules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SecuritycenterProjectsSecurityHealthAnalyticsSettingsEffectiveCustomModulesGetRequest', response_type_name='GoogleCloudSecuritycenterV1EffectiveSecurityHealthAnalyticsCustomModule', supports_download=False)

    def List(self, request, global_params=None):
        """Returns a list of all EffectiveSecurityHealthAnalyticsCustomModules for the given parent. This includes resident modules defined at the scope of the parent, and inherited modules, inherited from CRM ancestors.

      Args:
        request: (SecuritycenterProjectsSecurityHealthAnalyticsSettingsEffectiveCustomModulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListEffectiveSecurityHealthAnalyticsCustomModulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/securityHealthAnalyticsSettings/effectiveCustomModules', http_method='GET', method_id='securitycenter.projects.securityHealthAnalyticsSettings.effectiveCustomModules.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/effectiveCustomModules', request_field='', request_type_name='SecuritycenterProjectsSecurityHealthAnalyticsSettingsEffectiveCustomModulesListRequest', response_type_name='ListEffectiveSecurityHealthAnalyticsCustomModulesResponse', supports_download=False)