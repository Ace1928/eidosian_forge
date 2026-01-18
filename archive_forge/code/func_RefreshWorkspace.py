from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.source.v1 import source_v1_messages as messages
def RefreshWorkspace(self, request, global_params=None):
    """Brings a workspace up to date by merging in the changes made between its.
baseline and the revision to which its alias currently refers.
FAILED_PRECONDITION is returned if the alias refers to a revision that is
not a descendant of the workspace baseline, or if the workspace has no
baseline. Returns ABORTED when the workspace is simultaneously modified by
another client.

A refresh may involve merging files in the workspace with files in the
current alias revision. If this merge results in conflicts, then the
workspace is in a merge state: the merge_info field of Workspace will be
populated, and conflicting files in the workspace will contain conflict
markers.

      Args:
        request: (SourceProjectsReposWorkspacesRefreshWorkspaceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workspace) The response message.
      """
    config = self.GetMethodConfig('RefreshWorkspace')
    return self._RunMethod(config, request, global_params=global_params)