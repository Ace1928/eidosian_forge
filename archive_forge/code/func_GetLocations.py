from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v3beta import iam_v3beta_messages as messages
def GetLocations(self, request, global_params=None):
    """Gets information about a location.

      Args:
        request: (IamProjectsGetLocationsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudLocationLocation) The response message.
      """
    config = self.GetMethodConfig('GetLocations')
    return self._RunMethod(config, request, global_params=global_params)