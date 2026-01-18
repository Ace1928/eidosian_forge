from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.kms import crc32c
from googlecloudsdk.command_lib.kms import e2e_integrity
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def _VerifyResponseIntegrityFields(self, req, resp):
    """Verifies integrity fields in MacVerifyResponse."""
    if req.name != resp.name:
        raise e2e_integrity.ResourceNameVerificationError(e2e_integrity.GetResourceNameMismatchErrorMessage(req.name, resp.name))
    if not resp.verifiedDataCrc32c:
        raise e2e_integrity.ClientSideIntegrityVerificationError(e2e_integrity.GetRequestToServerCorruptedErrorMessage())
    if not resp.verifiedMacCrc32c:
        raise e2e_integrity.ClientSideIntegrityVerificationError(e2e_integrity.GetRequestToServerCorruptedErrorMessage())
    if resp.success != resp.verifiedSuccessIntegrity:
        raise e2e_integrity.ClientSideIntegrityVerificationError(e2e_integrity.GetResponseFromServerCorruptedErrorMessage())