from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import exceptions as kms_exceptions
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.command_lib.kms import maps
def CreateRequest(self, args, messages, fields_to_update):
    version_ref = flags.ParseCryptoKeyVersionName(args)
    req = messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsPatchRequest(name=version_ref.RelativeName(), cryptoKeyVersion=messages.CryptoKeyVersion(state=maps.CRYPTO_KEY_VERSION_STATE_MAPPER.GetEnumForChoice(args.state), externalProtectionLevelOptions=messages.ExternalProtectionLevelOptions(externalKeyUri=args.external_key_uri, ekmConnectionKeyPath=args.ekm_connection_key_path)))
    req.updateMask = ','.join(fields_to_update)
    return req