from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def DeleteData(self, request, global_params=None):
    """Deletes the data from a debug session. This does not cancel the debug session or prevent further data from being collected if the session is still active in runtime pods.

      Args:
        request: (ApigeeOrganizationsEnvironmentsApisRevisionsDebugsessionsDeleteDataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
    config = self.GetMethodConfig('DeleteData')
    return self._RunMethod(config, request, global_params=global_params)