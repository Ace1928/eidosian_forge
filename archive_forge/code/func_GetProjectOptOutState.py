from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudkms.v1alpha1 import cloudkms_v1alpha1_messages as messages
def GetProjectOptOutState(self, request, global_params=None):
    """Checks the project metadata and returns a boolean value indicating whether or not the project has been opted out. Fails with code.INVALID_ARGUMENT if the metadata type is unsupported or no longer valid (the related MSA notification period has expired).

      Args:
        request: (CloudkmsProjectsGetProjectOptOutStateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetProjectOptOutStateResponse) The response message.
      """
    config = self.GetMethodConfig('GetProjectOptOutState')
    return self._RunMethod(config, request, global_params=global_params)