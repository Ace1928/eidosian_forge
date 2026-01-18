from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firebasedataconnect.v1alpha import firebasedataconnect_v1alpha_messages as messages
def ExecuteGraphqlRead(self, request, global_params=None):
    """Execute any GraphQL query against the Firebase Data Connect's generated GraphQL schema. Grants full read to the connected data sources. `ExecuteGraphqlRead` is identical to `ExecuteGraphql` except it only accepts read-only query.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesExecuteGraphqlReadRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GraphqlResponse) The response message.
      """
    config = self.GetMethodConfig('ExecuteGraphqlRead')
    return self._RunMethod(config, request, global_params=global_params)