from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composer.v1alpha2 import composer_v1alpha2_messages as messages
def GetSourceCode(self, request, global_params=None):
    """Retrieves DAG source code.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsDagsGetSourceCodeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SourceCode) The response message.
      """
    config = self.GetMethodConfig('GetSourceCode')
    return self._RunMethod(config, request, global_params=global_params)