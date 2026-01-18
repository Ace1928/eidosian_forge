from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firebasedataconnect.v1alpha import firebasedataconnect_v1alpha_messages as messages
def ExecuteGraphql(self, request, global_params=None):
    """Execute any GraphQL query and mutation against the Firebase Data Connect's generated GraphQL schema. Grants full read and write access to the connected data sources. Note: Use introspection query to explore the generated GraphQL schema.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesExecuteGraphqlRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GraphqlResponse) The response message.
      """
    config = self.GetMethodConfig('ExecuteGraphql')
    return self._RunMethod(config, request, global_params=global_params)