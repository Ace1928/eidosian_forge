from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.redis.v1alpha1 import redis_v1alpha1_messages as messages
def ReportInstanceHealth(self, request, global_params=None):
    """Gets health report for an instance.

      Args:
        request: (RedisProjectsLocationsInstancesReportInstanceHealthRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReportInstanceHealthResponse) The response message.
      """
    config = self.GetMethodConfig('ReportInstanceHealth')
    return self._RunMethod(config, request, global_params=global_params)