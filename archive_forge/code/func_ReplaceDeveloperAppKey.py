from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def ReplaceDeveloperAppKey(self, request, global_params=None):
    """Updates the scope of an app. This API replaces the existing scopes with those specified in the request. Include or exclude any existing scopes that you want to retain or delete, respectively. The specified scopes must already be defined for the API products associated with the app. This API sets the `scopes` element under the `apiProducts` element in the attributes of the app.

      Args:
        request: (ApigeeOrganizationsDevelopersAppsKeysReplaceDeveloperAppKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperAppKey) The response message.
      """
    config = self.GetMethodConfig('ReplaceDeveloperAppKey')
    return self._RunMethod(config, request, global_params=global_params)