from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1 import securitycenter_v1_messages as messages
class FoldersEventThreatDetectionSettingsService(base_api.BaseApiService):
    """Service class for the folders_eventThreatDetectionSettings resource."""
    _NAME = 'folders_eventThreatDetectionSettings'

    def __init__(self, client):
        super(SecuritycenterV1.FoldersEventThreatDetectionSettingsService, self).__init__(client)
        self._upload_configs = {}

    def ValidateCustomModule(self, request, global_params=None):
        """Validates the given Event Threat Detection custom module.

      Args:
        request: (SecuritycenterFoldersEventThreatDetectionSettingsValidateCustomModuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ValidateEventThreatDetectionCustomModuleResponse) The response message.
      """
        config = self.GetMethodConfig('ValidateCustomModule')
        return self._RunMethod(config, request, global_params=global_params)
    ValidateCustomModule.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/eventThreatDetectionSettings:validateCustomModule', http_method='POST', method_id='securitycenter.folders.eventThreatDetectionSettings.validateCustomModule', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}:validateCustomModule', request_field='validateEventThreatDetectionCustomModuleRequest', request_type_name='SecuritycenterFoldersEventThreatDetectionSettingsValidateCustomModuleRequest', response_type_name='ValidateEventThreatDetectionCustomModuleResponse', supports_download=False)