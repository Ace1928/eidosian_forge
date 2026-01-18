from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def UpdateApiProxyRevision(self, request, global_params=None):
    """Updates an existing API proxy revision by uploading the API proxy configuration bundle as a zip file from your local machine. You can update only API proxy revisions that have never been deployed. After deployment, an API proxy revision becomes immutable, even if it is undeployed. Set the `Content-Type` header to either `multipart/form-data` or `application/octet-stream`.

      Args:
        request: (ApigeeOrganizationsApisRevisionsUpdateApiProxyRevisionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiProxyRevision) The response message.
      """
    config = self.GetMethodConfig('UpdateApiProxyRevision')
    return self._RunMethod(config, request, global_params=global_params)