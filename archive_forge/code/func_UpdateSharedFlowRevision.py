from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def UpdateSharedFlowRevision(self, request, global_params=None):
    """Updates a shared flow revision. This operation is only allowed on revisions which have never been deployed. After deployment a revision becomes immutable, even if it becomes undeployed. The payload is a ZIP-formatted shared flow. Content type must be either multipart/form-data or application/octet-stream.

      Args:
        request: (ApigeeOrganizationsSharedflowsRevisionsUpdateSharedFlowRevisionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SharedFlowRevision) The response message.
      """
    config = self.GetMethodConfig('UpdateSharedFlowRevision')
    return self._RunMethod(config, request, global_params=global_params)