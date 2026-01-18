from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
def MakeAdmin(self, request, global_params=None):
    """change admin status of a user.

      Args:
        request: (DirectoryUsersMakeAdminRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryUsersMakeAdminResponse) The response message.
      """
    config = self.GetMethodConfig('MakeAdmin')
    return self._RunMethod(config, request, global_params=global_params)