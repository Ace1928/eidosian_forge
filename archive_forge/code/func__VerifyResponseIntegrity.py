from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import e2e_integrity
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.core import log
def _VerifyResponseIntegrity(self, req, resp):
    """Verifies that the opt-out preference has been updated correctly.

    Args:
      req: messages.CloudkmsProjectsSetProjectOptOutStateRequest() object
      resp: messages.SetProjectOptOutStateResponse() object.

    Returns:
      Void.
    Raises:
      e2e_integrity.ClientSideIntegrityVerificationError if response integrity
      verification fails.
    """
    if bool(req.setProjectOptOutStateRequest.value) != bool(resp.value):
        raise e2e_integrity.ClientSideIntegrityVerificationError('Your opt-out preference could not be updated correctly. Please try again.')