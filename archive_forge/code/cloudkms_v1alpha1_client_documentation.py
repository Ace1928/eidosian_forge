from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudkms.v1alpha1 import cloudkms_v1alpha1_messages as messages
Updates the project metadata according to the new customer preference, and returns a boolean value to confirm the updated project metadata value. Fails with code.INVALID_ARGUMENT if the metadata type is unsupported or no longer valid (the related MSA notification period has expired).

      Args:
        request: (CloudkmsProjectsSetProjectOptOutStateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SetProjectOptOutStateResponse) The response message.
      