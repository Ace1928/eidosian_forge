from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def GenerateDownloadUrl(self, request, global_params=None):
    """Generates a signed URL for downloading the original zip file used to create an Archive Deployment. The URL is only valid for a limited period and should be used within minutes after generation. Each call returns a new upload URL.

      Args:
        request: (ApigeeOrganizationsEnvironmentsArchiveDeploymentsGenerateDownloadUrlRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1GenerateDownloadUrlResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateDownloadUrl')
    return self._RunMethod(config, request, global_params=global_params)