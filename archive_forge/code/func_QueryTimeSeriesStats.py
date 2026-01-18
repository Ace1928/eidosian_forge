from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def QueryTimeSeriesStats(self, request, global_params=None):
    """Retrieve security statistics as a collection of time series.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityStatsQueryTimeSeriesStatsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1QueryTimeSeriesStatsResponse) The response message.
      """
    config = self.GetMethodConfig('QueryTimeSeriesStats')
    return self._RunMethod(config, request, global_params=global_params)