from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1 import securitycenter_v1_messages as messages
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