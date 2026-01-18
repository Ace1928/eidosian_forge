from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v1 import monitoring_v1_messages as messages
def QueryExemplars(self, request, global_params=None):
    """Lists exemplars relevant to a given PromQL query,.

      Args:
        request: (MonitoringProjectsLocationPrometheusApiV1QueryExemplarsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
    config = self.GetMethodConfig('QueryExemplars')
    return self._RunMethod(config, request, global_params=global_params)