from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.oslogin.v1beta import oslogin_v1beta_messages as messages
def GetLoginProfile(self, request, global_params=None):
    """Retrieves the profile information used for logging in to a virtual machine on Google Compute Engine.

      Args:
        request: (OsloginUsersGetLoginProfileRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LoginProfile) The response message.
      """
    config = self.GetMethodConfig('GetLoginProfile')
    return self._RunMethod(config, request, global_params=global_params)