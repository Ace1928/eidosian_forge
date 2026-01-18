from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firebasedataconnect.v1alpha import firebasedataconnect_v1alpha_messages as messages
def ExecuteQuery(self, request, global_params=None):
    """Execute a predefined query in a Connector.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesConnectorsExecuteQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExecuteQueryResponse) The response message.
      """
    config = self.GetMethodConfig('ExecuteQuery')
    return self._RunMethod(config, request, global_params=global_params)