from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def UpdateAppGroupAppKey(self, request, global_params=None):
    """Adds an API product to an AppGroupAppKey, enabling the app that holds the key to access the API resources bundled in the API product. In addition, you can add attributes to the AppGroupAppKey. This API replaces the existing attributes with those specified in the request. Include or exclude any existing attributes that you want to retain or delete, respectively. You can use the same key to access all API products associated with the app.

      Args:
        request: (ApigeeOrganizationsAppgroupsAppsKeysUpdateAppGroupAppKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1AppGroupAppKey) The response message.
      """
    config = self.GetMethodConfig('UpdateAppGroupAppKey')
    return self._RunMethod(config, request, global_params=global_params)