from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
def SearchJobs(self, request, global_params=None):
    """Obtain list of spark jobs corresponding to a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSearchJobsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchSessionSparkApplicationJobsResponse) The response message.
      """
    config = self.GetMethodConfig('SearchJobs')
    return self._RunMethod(config, request, global_params=global_params)