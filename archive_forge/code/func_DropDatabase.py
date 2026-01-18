from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
def DropDatabase(self, request, global_params=None):
    """Drops (aka deletes) a Cloud Spanner database. Completed backups for the database will be retained according to their `expire_time`. Note: Cloud Spanner might continue to accept requests for a few seconds after the database has been deleted.

      Args:
        request: (SpannerProjectsInstancesDatabasesDropDatabaseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('DropDatabase')
    return self._RunMethod(config, request, global_params=global_params)