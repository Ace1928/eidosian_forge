from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.testing.v1 import testing_v1_messages as messages
def GetApkDetails(self, request, global_params=None):
    """Gets the details of an Android application APK.

      Args:
        request: (TestingApplicationDetailServiceGetApkDetailsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetApkDetailsResponse) The response message.
      """
    config = self.GetMethodConfig('GetApkDetails')
    return self._RunMethod(config, request, global_params=global_params)