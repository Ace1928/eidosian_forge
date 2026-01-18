from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
def DeleteSnapshots(self, request, global_params=None):
    """Deletes a snapshot.

      Args:
        request: (DataflowProjectsDeleteSnapshotsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DeleteSnapshotResponse) The response message.
      """
    config = self.GetMethodConfig('DeleteSnapshots')
    return self._RunMethod(config, request, global_params=global_params)