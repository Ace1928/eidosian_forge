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
from googlecloudsdk.core.util import files
def _CreateEncryptRequest(self, args):
    if args.plaintext_file == '-' and args.additional_authenticated_data_file == '-':
        raise exceptions.InvalidArgumentException('--plaintext-file', '--plaintext-file and --additional-authenticated-data-file cannot both read from stdin.')
    try:
        plaintext = self._ReadFileOrStdin(args.plaintext_file, max_bytes=65536)
    except files.Error as e:
        raise exceptions.BadFileException('Failed to read plaintext file [{0}]: {1}'.format(args.plaintext_file, e))
    aad = None
    if args.additional_authenticated_data_file:
        try:
            aad = self._ReadFileOrStdin(args.additional_authenticated_data_file, max_bytes=65536)
        except files.Error as e:
            raise exceptions.BadFileException('Failed to read additional authenticated data file [{0}]: {1}'.format(args.additional_authenticated_data_file, e))
    if args.version:
        crypto_key_ref = flags.ParseCryptoKeyVersionName(args)
    else:
        crypto_key_ref = flags.ParseCryptoKeyName(args)
    messages = cloudkms_base.GetMessagesModule()
    req = messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysEncryptRequest(name=crypto_key_ref.RelativeName())
    if self._PerformIntegrityVerification(args):
        plaintext_crc32c = crc32c.Crc32c(plaintext)
        aad_crc32c = crc32c.Crc32c(aad) if aad is not None else crc32c.Crc32c(b'')
        req.encryptRequest = messages.EncryptRequest(plaintext=plaintext, additionalAuthenticatedData=aad, plaintextCrc32c=plaintext_crc32c, additionalAuthenticatedDataCrc32c=aad_crc32c)
    else:
        req.encryptRequest = messages.EncryptRequest(plaintext=plaintext, additionalAuthenticatedData=aad)
    return req