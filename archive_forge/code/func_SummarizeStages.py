from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
def SummarizeStages(self, request, global_params=None):
    """Obtain summary of Stages for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSummarizeStagesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SummarizeSessionSparkApplicationStagesResponse) The response message.
      """
    config = self.GetMethodConfig('SummarizeStages')
    return self._RunMethod(config, request, global_params=global_params)