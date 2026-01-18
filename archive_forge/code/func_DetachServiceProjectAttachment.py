from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apphub.v1alpha import apphub_v1alpha_messages as messages
def DetachServiceProjectAttachment(self, request, global_params=None):
    """Detaches a service project from a host project. You can call this API from any service project without needing access to the host project that it is attached to.

      Args:
        request: (ApphubProjectsLocationsDetachServiceProjectAttachmentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DetachServiceProjectAttachmentResponse) The response message.
      """
    config = self.GetMethodConfig('DetachServiceProjectAttachment')
    return self._RunMethod(config, request, global_params=global_params)