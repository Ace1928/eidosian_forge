from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.toolresults.v1beta3 import toolresults_v1beta3_messages as messages
def GetPerfMetricsSummary(self, request, global_params=None):
    """Retrieves a PerfMetricsSummary. May return any of the following error code(s): - NOT_FOUND - The specified PerfMetricsSummary does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsStepsGetPerfMetricsSummaryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PerfMetricsSummary) The response message.
      """
    config = self.GetMethodConfig('GetPerfMetricsSummary')
    return self._RunMethod(config, request, global_params=global_params)