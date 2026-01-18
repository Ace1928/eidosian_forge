from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def UpdateEnvironment(self, request, global_params=None):
    """Updates an existing environment. When updating properties, you must pass all existing properties to the API, even if they are not being changed. If you omit properties from the payload, the properties are removed. To get the current list of properties for the environment, use the [Get Environment API](get). **Note**: Both `PUT` and `POST` methods are supported for updating an existing environment.

      Args:
        request: (GoogleCloudApigeeV1Environment) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Environment) The response message.
      """
    config = self.GetMethodConfig('UpdateEnvironment')
    return self._RunMethod(config, request, global_params=global_params)