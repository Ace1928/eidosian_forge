from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
def GetScans(self, request, global_params=None):
    """Request a specific scan with Database-specific data for Cloud Key Visualizer.

      Args:
        request: (SpannerProjectsInstancesDatabasesGetScansRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Scan) The response message.
      """
    config = self.GetMethodConfig('GetScans')
    return self._RunMethod(config, request, global_params=global_params)