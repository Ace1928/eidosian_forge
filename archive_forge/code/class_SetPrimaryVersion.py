from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.command_lib.kms import resource_args
class SetPrimaryVersion(base.Command):
    """Set the primary version of a key.

  Sets the specified version as the primary version of the given key.
  The version is specified by its version number assigned on creation.

  ## EXAMPLES

  The following command sets version 9 as the primary version of the
  key `samwise` within keyring `fellowship` and location `global`:

    $ {command} samwise --version=9 --keyring=fellowship --location=global
  """

    @staticmethod
    def Args(parser):
        resource_args.AddKmsKeyResourceArgForKMS(parser, True, 'key')
        flags.AddCryptoKeyVersionFlag(parser, 'to make primary', required=True)

    def Run(self, args):
        client = cloudkms_base.GetClientInstance()
        messages = cloudkms_base.GetMessagesModule()
        version_ref = flags.ParseCryptoKeyVersionName(args)
        key_ref = flags.ParseCryptoKeyName(args)
        req = messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysUpdatePrimaryVersionRequest(name=key_ref.RelativeName(), updateCryptoKeyPrimaryVersionRequest=messages.UpdateCryptoKeyPrimaryVersionRequest(cryptoKeyVersionId=version_ref.cryptoKeyVersionsId))
        return client.projects_locations_keyRings_cryptoKeys.UpdatePrimaryVersion(req)