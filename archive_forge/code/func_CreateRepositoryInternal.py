from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securesourcemanager.v1 import securesourcemanager_v1_messages as messages
def CreateRepositoryInternal(self, request, global_params=None):
    """THIS METHOD IS FOR INTERNAL USE ONLY.

      Args:
        request: (SecuresourcemanagerProjectsLocationsRepositoriesCreateRepositoryInternalRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Repository) The response message.
      """
    config = self.GetMethodConfig('CreateRepositoryInternal')
    return self._RunMethod(config, request, global_params=global_params)