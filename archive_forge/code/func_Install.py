from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gsuiteaddons.v1 import gsuiteaddons_v1_messages as messages
def Install(self, request, global_params=None):
    """Installs a deployment to your account for testing. For more information, see [Test your add-on](https://developers.google.com/workspace/add-ons/guides/alternate-runtimes#test_your_add-on).

      Args:
        request: (GsuiteaddonsProjectsDeploymentsInstallRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
    config = self.GetMethodConfig('Install')
    return self._RunMethod(config, request, global_params=global_params)