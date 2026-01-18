from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v1 import monitoring_v1_messages as messages
def QueryRange(self, request, global_params=None):
    """Evaluate a PromQL query with start, end time range.

      Args:
        request: (MonitoringProjectsLocationPrometheusApiV1QueryRangeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
    config = self.GetMethodConfig('QueryRange')
    return self._RunMethod(config, request, global_params=global_params)