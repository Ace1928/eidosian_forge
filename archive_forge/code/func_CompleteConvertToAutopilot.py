from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
def CompleteConvertToAutopilot(self, request, global_params=None):
    """CompleteConvertToAutopilot is an optional API that commits the conversion by deleting all Standard node pools and completing CA rotation. This action requires that a conversion has been started and that workload migration has completed, with no pods running on GKE Standard node pools. This action will be automatically performed 72 hours after conversion.

      Args:
        request: (ContainerProjectsLocationsClustersCompleteConvertToAutopilotRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('CompleteConvertToAutopilot')
    return self._RunMethod(config, request, global_params=global_params)