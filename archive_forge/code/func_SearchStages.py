from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
def SearchStages(self, request, global_params=None):
    """Obtain data corresponding to stages for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSearchStagesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchSessionSparkApplicationStagesResponse) The response message.
      """
    config = self.GetMethodConfig('SearchStages')
    return self._RunMethod(config, request, global_params=global_params)