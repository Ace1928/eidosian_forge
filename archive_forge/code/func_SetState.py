from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
def SetState(self, request, global_params=None):
    """Updates the state of a finding. If no location is specified, finding is assumed to be in global.

      Args:
        request: (SecuritycenterProjectsSourcesLocationsFindingsSetStateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV2Finding) The response message.
      """
    config = self.GetMethodConfig('SetState')
    return self._RunMethod(config, request, global_params=global_params)