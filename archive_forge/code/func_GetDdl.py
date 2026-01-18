from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
def GetDdl(self, request, global_params=None):
    """Returns the schema of a Cloud Spanner database as a list of formatted DDL statements. This method does not show pending schema updates, those may be queried using the Operations API.

      Args:
        request: (SpannerProjectsInstancesDatabasesGetDdlRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetDatabaseDdlResponse) The response message.
      """
    config = self.GetMethodConfig('GetDdl')
    return self._RunMethod(config, request, global_params=global_params)