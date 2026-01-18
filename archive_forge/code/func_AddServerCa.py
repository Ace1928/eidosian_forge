from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
def AddServerCa(self, request, global_params=None):
    """Add a new trusted Certificate Authority (CA) version for the specified instance. Required to prepare for a certificate rotation. If a CA version was previously added but never used in a certificate rotation, this operation replaces that version. There cannot be more than one CA version waiting to be rotated in.

      Args:
        request: (SqlInstancesAddServerCaRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('AddServerCa')
    return self._RunMethod(config, request, global_params=global_params)