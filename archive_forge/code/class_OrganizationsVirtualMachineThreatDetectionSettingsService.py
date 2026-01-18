from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1beta2 import securitycenter_v1beta2_messages as messages
class OrganizationsVirtualMachineThreatDetectionSettingsService(base_api.BaseApiService):
    """Service class for the organizations_virtualMachineThreatDetectionSettings resource."""
    _NAME = 'organizations_virtualMachineThreatDetectionSettings'

    def __init__(self, client):
        super(SecuritycenterV1beta2.OrganizationsVirtualMachineThreatDetectionSettingsService, self).__init__(client)
        self._upload_configs = {}

    def Calculate(self, request, global_params=None):
        """Calculates the effective VirtualMachineThreatDetectionSettings based on its level in the resource hierarchy and its settings. Settings provided closer to the target resource take precedence over those further away (e.g. folder will override organization level settings). The default SCC setting for the detector service defaults can be overridden at organization, folder and project levels. No assumptions should be made about the SCC defaults as it is considered an internal implementation detail.

      Args:
        request: (SecuritycenterOrganizationsVirtualMachineThreatDetectionSettingsCalculateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VirtualMachineThreatDetectionSettings) The response message.
      """
        config = self.GetMethodConfig('Calculate')
        return self._RunMethod(config, request, global_params=global_params)
    Calculate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta2/organizations/{organizationsId}/virtualMachineThreatDetectionSettings:calculate', http_method='GET', method_id='securitycenter.organizations.virtualMachineThreatDetectionSettings.calculate', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta2/{+name}:calculate', request_field='', request_type_name='SecuritycenterOrganizationsVirtualMachineThreatDetectionSettingsCalculateRequest', response_type_name='VirtualMachineThreatDetectionSettings', supports_download=False)