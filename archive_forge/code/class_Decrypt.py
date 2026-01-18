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
class Decrypt(base.Command):
    """Decrypt a ciphertext file using a Cloud KMS key.

  `{command}` decrypts the given ciphertext file using the given Cloud KMS key
  and writes the result to the named plaintext file. Note that to permit users
  to decrypt using a key, they must be have at least one of the following IAM
  roles for that key: `roles/cloudkms.cryptoKeyDecrypter`,
  `roles/cloudkms.cryptoKeyEncrypterDecrypter`.

  Additional authenticated data (AAD) is used as an additional check by Cloud
  KMS to authenticate a decryption request. If an additional authenticated data
  file is provided, its contents must match the additional authenticated data
  provided during encryption and must not be larger than 64KiB. If you don't
  provide a value for `--additional-authenticated-data-file`, an empty string is
  used. For a thorough explanation of AAD, refer to this
  guide: https://cloud.google.com/kms/docs/additional-authenticated-data

  If `--ciphertext-file` or `--additional-authenticated-data-file` is set to
  '-', that file is read from stdin. Note that both files cannot be read from
  stdin. Similarly, if `--plaintext-file` is set to '-', the decrypted plaintext
  is written to stdout.

  By default, the command performs integrity verification on data sent to and
  received from Cloud KMS. Use `--skip-integrity-verification` to disable
  integrity verification.

  ## EXAMPLES

  To decrypt the file 'path/to/ciphertext' using the key `frodo` with key
  ring `fellowship` and location `global` and write the plaintext
  to 'path/to/plaintext.dec', run:

    $ {command} \\
        --key=frodo \\
        --keyring=fellowship \\
        --location=global \\
        --ciphertext-file=path/to/input/ciphertext \\
        --plaintext-file=path/to/output/plaintext.dec

  To decrypt the file 'path/to/ciphertext' using the key `frodo` and the
  additional authenticated data that was used to encrypt the ciphertext, and
  write the decrypted plaintext to stdout, run:

    $ {command} \\
        --key=frodo \\
        --keyring=fellowship \\
        --location=global \\
        --additional-authenticated-data-file=path/to/aad \\
        --ciphertext-file=path/to/input/ciphertext \\
        --plaintext-file='-'
  """

    @staticmethod
    def Args(parser):
        flags.AddKeyResourceFlags(parser, "Cloud KMS key to use for decryption.\n* For symmetric keys, Cloud KMS detects the decryption key version from the ciphertext. If you specify a key version as part of a symmetric decryption request, an error is logged and decryption fails.\n* For asymmetric keys, the encryption key version can't be detected automatically. You must keep track of this information and provide the key version in the decryption request. The key version itself is not sensitive data and does not need to be encrypted.")
        flags.AddCiphertextFileFlag(parser, 'to decrypt. This file should contain the result of encrypting a file with `gcloud kms encrypt`')
        flags.AddPlaintextFileFlag(parser, 'to output')
        flags.AddAadFileFlag(parser)
        flags.AddSkipIntegrityVerification(parser)

    def _ReadFileOrStdin(self, path, max_bytes):
        data = console_io.ReadFromFileOrStdin(path, binary=True)
        if len(data) > max_bytes:
            raise exceptions.BadFileException('The file [{0}] is larger than the maximum size of {1} bytes.'.format(path, max_bytes))
        return data

    def _PerformIntegrityVerification(self, args):
        return not args.skip_integrity_verification

    def _CreateDecryptRequest(self, args):
        if args.ciphertext_file == '-' and args.additional_authenticated_data_file == '-':
            raise exceptions.InvalidArgumentException('--ciphertext-file', '--ciphertext-file and --additional-authenticated-data-file cannot both read from stdin.')
        try:
            ciphertext = self._ReadFileOrStdin(args.ciphertext_file, max_bytes=2 * 65536)
        except files.Error as e:
            raise exceptions.BadFileException('Failed to read ciphertext file [{0}]: {1}'.format(args.ciphertext_file, e))
        aad = None
        if args.additional_authenticated_data_file:
            try:
                aad = self._ReadFileOrStdin(args.additional_authenticated_data_file, max_bytes=65536)
            except files.Error as e:
                raise exceptions.BadFileException('Failed to read additional authenticated data file [{0}]: {1}'.format(args.additional_authenticated_data_file, e))
        crypto_key_ref = flags.ParseCryptoKeyName(args)
        if '/cryptoKeyVersions/' in crypto_key_ref.cryptoKeysId:
            raise exceptions.InvalidArgumentException('--key', '{} includes cryptoKeyVersion which is not valid for decrypt.'.format(crypto_key_ref.cryptoKeysId))
        messages = cloudkms_base.GetMessagesModule()
        req = messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysDecryptRequest(name=crypto_key_ref.RelativeName())
        if self._PerformIntegrityVerification(args):
            ciphertext_crc32c = crc32c.Crc32c(ciphertext)
            aad_crc32c = crc32c.Crc32c(aad) if aad is not None else crc32c.Crc32c(b'')
            req.decryptRequest = messages.DecryptRequest(ciphertext=ciphertext, additionalAuthenticatedData=aad, ciphertextCrc32c=ciphertext_crc32c, additionalAuthenticatedDataCrc32c=aad_crc32c)
        else:
            req.decryptRequest = messages.DecryptRequest(ciphertext=ciphertext, additionalAuthenticatedData=aad)
        return req

    def _VerifyResponseIntegrityFields(self, req, resp):
        """Verifies integrity fields in response."""
        if not crc32c.Crc32cMatches(resp.plaintext, resp.plaintextCrc32c):
            raise e2e_integrity.ClientSideIntegrityVerificationError(e2e_integrity.GetResponseFromServerCorruptedErrorMessage())

    def Run(self, args):
        req = self._CreateDecryptRequest(args)
        client = cloudkms_base.GetClientInstance()
        try:
            resp = client.projects_locations_keyRings_cryptoKeys.Decrypt(req)
        except apitools_exceptions.HttpBadRequestError as error:
            e2e_integrity.ProcessHttpBadRequestError(error)
        if self._PerformIntegrityVerification(args):
            self._VerifyResponseIntegrityFields(req, resp)
        try:
            if resp.plaintext is None:
                with files.FileWriter(args.plaintext_file):
                    pass
                log.Print('Decrypted file is empty')
            else:
                log.WriteToFileOrStdout(args.plaintext_file, resp.plaintext, binary=True, overwrite=True)
        except files.Error as e:
            raise exceptions.BadFileException(e)