from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
def SearchExecutorStageSummary(self, request, global_params=None):
    """Obtain executor summary with respect to a spark stage attempt.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSearchExecutorStageSummaryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchSessionSparkApplicationExecutorStageSummaryResponse) The response message.
      """
    config = self.GetMethodConfig('SearchExecutorStageSummary')
    return self._RunMethod(config, request, global_params=global_params)