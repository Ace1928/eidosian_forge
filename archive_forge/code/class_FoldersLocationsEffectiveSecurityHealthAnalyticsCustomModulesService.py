from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycentermanagement.v1 import securitycentermanagement_v1_messages as messages
class FoldersLocationsEffectiveSecurityHealthAnalyticsCustomModulesService(base_api.BaseApiService):
    """Service class for the folders_locations_effectiveSecurityHealthAnalyticsCustomModules resource."""
    _NAME = 'folders_locations_effectiveSecurityHealthAnalyticsCustomModules'

    def __init__(self, client):
        super(SecuritycentermanagementV1.FoldersLocationsEffectiveSecurityHealthAnalyticsCustomModulesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details of a single EffectiveSecurityHealthAnalyticsCustomModule.

      Args:
        request: (SecuritycentermanagementFoldersLocationsEffectiveSecurityHealthAnalyticsCustomModulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EffectiveSecurityHealthAnalyticsCustomModule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/locations/{locationsId}/effectiveSecurityHealthAnalyticsCustomModules/{effectiveSecurityHealthAnalyticsCustomModulesId}', http_method='GET', method_id='securitycentermanagement.folders.locations.effectiveSecurityHealthAnalyticsCustomModules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SecuritycentermanagementFoldersLocationsEffectiveSecurityHealthAnalyticsCustomModulesGetRequest', response_type_name='EffectiveSecurityHealthAnalyticsCustomModule', supports_download=False)

    def List(self, request, global_params=None):
        """Returns a list of all EffectiveSecurityHealthAnalyticsCustomModules for the given parent. This includes resident modules defined at the scope of the parent, and inherited modules, inherited from CRM ancestors (no descendants).

      Args:
        request: (SecuritycentermanagementFoldersLocationsEffectiveSecurityHealthAnalyticsCustomModulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListEffectiveSecurityHealthAnalyticsCustomModulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/locations/{locationsId}/effectiveSecurityHealthAnalyticsCustomModules', http_method='GET', method_id='securitycentermanagement.folders.locations.effectiveSecurityHealthAnalyticsCustomModules.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/effectiveSecurityHealthAnalyticsCustomModules', request_field='', request_type_name='SecuritycentermanagementFoldersLocationsEffectiveSecurityHealthAnalyticsCustomModulesListRequest', response_type_name='ListEffectiveSecurityHealthAnalyticsCustomModulesResponse', supports_download=False)