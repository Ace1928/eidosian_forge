from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
def SubmitAsOperation(self, request, global_params=None):
    """Submits job to a cluster.

      Args:
        request: (DataprocProjectsRegionsJobsSubmitAsOperationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SubmitAsOperation')
    return self._RunMethod(config, request, global_params=global_params)