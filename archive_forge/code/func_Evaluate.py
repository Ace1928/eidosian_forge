from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.binaryauthorization.v1 import binaryauthorization_v1_messages as messages
def Evaluate(self, request, global_params=None):
    """Evaluates a Kubernetes object versus a GKE platform policy. Returns `NOT_FOUND` if the policy doesn't exist, `INVALID_ARGUMENT` if the policy or request is malformed and `PERMISSION_DENIED` if the client does not have sufficient permissions.

      Args:
        request: (BinaryauthorizationProjectsPlatformsGkePoliciesEvaluateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EvaluateGkePolicyResponse) The response message.
      """
    config = self.GetMethodConfig('Evaluate')
    return self._RunMethod(config, request, global_params=global_params)