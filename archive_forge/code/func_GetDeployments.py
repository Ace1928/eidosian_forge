from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def GetDeployments(self, request, global_params=None):
    """Gets the deployment of a shared flow revision and actual state reported by runtime pods.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSharedflowsRevisionsGetDeploymentsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Deployment) The response message.
      """
    config = self.GetMethodConfig('GetDeployments')
    return self._RunMethod(config, request, global_params=global_params)