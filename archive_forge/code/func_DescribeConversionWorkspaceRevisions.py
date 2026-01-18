from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datamigration.v1 import datamigration_v1_messages as messages
def DescribeConversionWorkspaceRevisions(self, request, global_params=None):
    """Retrieves a list of committed revisions of a specific conversion workspace.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesDescribeConversionWorkspaceRevisionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DescribeConversionWorkspaceRevisionsResponse) The response message.
      """
    config = self.GetMethodConfig('DescribeConversionWorkspaceRevisions')
    return self._RunMethod(config, request, global_params=global_params)