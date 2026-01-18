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
from googlecloudsdk.command_lib.kms import get_digest
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def _CreateAsymmetricSignRequestOnData(self, args):
    """Returns an AsymmetricSignRequest for use with a data input.

    Populates an AsymmetricSignRequest with its data field populated by data
    read from args.input_file. dataCrc32c is populated if integrity verification
    is not skipped.

    Args:
      args: Input arguments.

    Returns:
      An AsymmetricSignRequest with data populated and dataCrc32c populated if
      integrity verification is not skipped.

    Raises:
      exceptions.BadFileException: An error occurred reading the input file.
      This can occur if the file can't be read or if the file is larger than
      64 KiB.
    """
    try:
        data = self._ReadBinaryFile(args.input_file, max_bytes=65536)
    except files.Error as e:
        raise exceptions.BadFileException('Failed to read input file [{0}]: {1}'.format(args.input_file, e))
    messages = cloudkms_base.GetMessagesModule()
    req = messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsAsymmetricSignRequest(name=flags.ParseCryptoKeyVersionName(args).RelativeName())
    if self._PerformIntegrityVerification(args):
        data_crc32c = crc32c.Crc32c(data)
        req.asymmetricSignRequest = messages.AsymmetricSignRequest(data=data, dataCrc32c=data_crc32c)
    else:
        req.asymmetricSignRequest = messages.AsymmetricSignRequest(data=data)
    return req