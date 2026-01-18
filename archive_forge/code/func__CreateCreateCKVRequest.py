from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import exceptions as kms_exceptions
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.core import log
def _CreateCreateCKVRequest(self, args):
    messages = cloudkms_base.GetMessagesModule()
    crypto_key_ref = flags.ParseCryptoKeyName(args)
    if args.external_key_uri and args.ekm_connection_key_path:
        raise kms_exceptions.ArgumentError('Can not specify both --external-key-uri and --ekm-connection-key-path.')
    if args.external_key_uri or args.ekm_connection_key_path:
        return messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsCreateRequest(parent=crypto_key_ref.RelativeName(), cryptoKeyVersion=messages.CryptoKeyVersion(externalProtectionLevelOptions=messages.ExternalProtectionLevelOptions(externalKeyUri=args.external_key_uri, ekmConnectionKeyPath=args.ekm_connection_key_path)))
    return messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsCreateRequest(parent=crypto_key_ref.RelativeName())