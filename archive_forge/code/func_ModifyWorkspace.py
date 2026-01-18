from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.source.v1 import source_v1_messages as messages
def ModifyWorkspace(self, request, global_params=None):
    """Applies an ordered sequence of file modification actions to a workspace.
Returns ABORTED if current_snapshot_id in the request does not refer to
the most recent update to the workspace or if the workspace is
simultaneously modified by another client.

      Args:
        request: (SourceProjectsReposWorkspacesModifyWorkspaceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workspace) The response message.
      """
    config = self.GetMethodConfig('ModifyWorkspace')
    return self._RunMethod(config, request, global_params=global_params)