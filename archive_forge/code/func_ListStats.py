from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composer.v1alpha2 import composer_v1alpha2_messages as messages
def ListStats(self, request, global_params=None):
    """List DAGs with statistics for a given time interval.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsDagsListStatsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDagStatsResponse) The response message.
      """
    config = self.GetMethodConfig('ListStats')
    return self._RunMethod(config, request, global_params=global_params)