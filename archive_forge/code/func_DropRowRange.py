from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigtableadmin.v2 import bigtableadmin_v2_messages as messages
def DropRowRange(self, request, global_params=None):
    """Permanently drop/delete a row range from a specified table. The request can specify whether to delete all rows in a table, or only those that match a particular prefix. Note that row key prefixes used here are treated as service data. For more information about how service data is handled, see the [Google Cloud Privacy Notice](https://cloud.google.com/terms/cloud-privacy-notice).

      Args:
        request: (BigtableadminProjectsInstancesTablesDropRowRangeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('DropRowRange')
    return self._RunMethod(config, request, global_params=global_params)