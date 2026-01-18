from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datamigration.v1 import datamigration_v1_messages as messages
def SearchBackgroundJobs(self, request, global_params=None):
    """Searches/lists the background jobs for a specific conversion workspace. The background jobs are not resources like conversion workspaces or mapping rules, and they can't be created, updated or deleted. Instead, they are a way to expose the data plane jobs log.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesSearchBackgroundJobsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchBackgroundJobsResponse) The response message.
      """
    config = self.GetMethodConfig('SearchBackgroundJobs')
    return self._RunMethod(config, request, global_params=global_params)