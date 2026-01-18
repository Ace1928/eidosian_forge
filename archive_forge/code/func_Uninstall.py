from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gsuiteaddons.v1 import gsuiteaddons_v1_messages as messages
def Uninstall(self, request, global_params=None):
    """Uninstalls a test deployment from the user's account. For more information, see [Test your add-on](https://developers.google.com/workspace/add-ons/guides/alternate-runtimes#test_your_add-on).

      Args:
        request: (GsuiteaddonsProjectsDeploymentsUninstallRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
    config = self.GetMethodConfig('Uninstall')
    return self._RunMethod(config, request, global_params=global_params)