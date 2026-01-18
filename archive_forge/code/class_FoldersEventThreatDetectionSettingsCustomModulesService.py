from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1 import securitycenter_v1_messages as messages
class FoldersEventThreatDetectionSettingsCustomModulesService(base_api.BaseApiService):
    """Service class for the folders_eventThreatDetectionSettings_customModules resource."""
    _NAME = 'folders_eventThreatDetectionSettings_customModules'

    def __init__(self, client):
        super(SecuritycenterV1.FoldersEventThreatDetectionSettingsCustomModulesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a resident Event Threat Detection custom module at the scope of the given Resource Manager parent, and also creates inherited custom modules for all descendants of the given parent. These modules are enabled by default.

      Args:
        request: (SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EventThreatDetectionCustomModule) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/eventThreatDetectionSettings/customModules', http_method='POST', method_id='securitycenter.folders.eventThreatDetectionSettings.customModules.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/customModules', request_field='eventThreatDetectionCustomModule', request_type_name='SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesCreateRequest', response_type_name='EventThreatDetectionCustomModule', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified Event Threat Detection custom module and all of its descendants in the Resource Manager hierarchy. This method is only supported for resident custom modules.

      Args:
        request: (SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/eventThreatDetectionSettings/customModules/{customModulesId}', http_method='DELETE', method_id='securitycenter.folders.eventThreatDetectionSettings.customModules.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an Event Threat Detection custom module.

      Args:
        request: (SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EventThreatDetectionCustomModule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/eventThreatDetectionSettings/customModules/{customModulesId}', http_method='GET', method_id='securitycenter.folders.eventThreatDetectionSettings.customModules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesGetRequest', response_type_name='EventThreatDetectionCustomModule', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all Event Threat Detection custom modules for the given Resource Manager parent. This includes resident modules defined at the scope of the parent along with modules inherited from ancestors.

      Args:
        request: (SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListEventThreatDetectionCustomModulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/eventThreatDetectionSettings/customModules', http_method='GET', method_id='securitycenter.folders.eventThreatDetectionSettings.customModules.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/customModules', request_field='', request_type_name='SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesListRequest', response_type_name='ListEventThreatDetectionCustomModulesResponse', supports_download=False)

    def ListDescendant(self, request, global_params=None):
        """Lists all resident Event Threat Detection custom modules under the given Resource Manager parent and its descendants.

      Args:
        request: (SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesListDescendantRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDescendantEventThreatDetectionCustomModulesResponse) The response message.
      """
        config = self.GetMethodConfig('ListDescendant')
        return self._RunMethod(config, request, global_params=global_params)
    ListDescendant.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/eventThreatDetectionSettings/customModules:listDescendant', http_method='GET', method_id='securitycenter.folders.eventThreatDetectionSettings.customModules.listDescendant', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/customModules:listDescendant', request_field='', request_type_name='SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesListDescendantRequest', response_type_name='ListDescendantEventThreatDetectionCustomModulesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the Event Threat Detection custom module with the given name based on the given update mask. Updating the enablement state is supported for both resident and inherited modules (though resident modules cannot have an enablement state of "inherited"). Updating the display name or configuration of a module is supported for resident modules only. The type of a module cannot be changed.

      Args:
        request: (SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EventThreatDetectionCustomModule) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/eventThreatDetectionSettings/customModules/{customModulesId}', http_method='PATCH', method_id='securitycenter.folders.eventThreatDetectionSettings.customModules.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='eventThreatDetectionCustomModule', request_type_name='SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesPatchRequest', response_type_name='EventThreatDetectionCustomModule', supports_download=False)