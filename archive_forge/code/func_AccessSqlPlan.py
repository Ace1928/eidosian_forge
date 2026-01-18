from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
def AccessSqlPlan(self, request, global_params=None):
    """Obtain Spark Plan Graph for a Spark Application SQL execution. Limits the number of clusters returned as part of the graph to 10000.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsAccessSqlPlanRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AccessSessionSparkApplicationSqlSparkPlanGraphResponse) The response message.
      """
    config = self.GetMethodConfig('AccessSqlPlan')
    return self._RunMethod(config, request, global_params=global_params)