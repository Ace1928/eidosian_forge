from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composer.v1alpha2 import composer_v1alpha2_messages as messages
def LoadSnapshot(self, request, global_params=None):
    """Loads a snapshot of a Cloud Composer environment. As a result of this operation, a snapshot of environment's specified in LoadSnapshotRequest is loaded into the environment.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsLoadSnapshotRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('LoadSnapshot')
    return self._RunMethod(config, request, global_params=global_params)