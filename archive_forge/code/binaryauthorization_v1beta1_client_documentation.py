from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.binaryauthorization.v1beta1 import binaryauthorization_v1beta1_messages as messages
Gets the current system policy in the specified location.

      Args:
        request: (BinaryauthorizationSystempolicyGetPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      