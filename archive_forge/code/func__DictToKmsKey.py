from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _DictToKmsKey(args):
    """Returns the Cloud KMS crypto key name based on the KMS args."""
    if not args:
        return None

    def GetValue(args, key):

        def GetValueFunc():
            val = args[key] if key in args else None
            if val:
                return val
            raise calliope_exceptions.InvalidArgumentException('--create-disk', 'KMS cryptokey resource was not fully specified. Key [{}] must be specified.'.format(key))
        return GetValueFunc
    return resources.REGISTRY.Parse(GetValue(args, 'kms-key')(), params={'projectsId': args['kms-project'] if 'kms-project' in args else properties.VALUES.core.project.GetOrFail, 'locationsId': GetValue(args, 'kms-location'), 'keyRingsId': GetValue(args, 'kms-keyring'), 'cryptoKeysId': GetValue(args, 'kms-key')}, collection='cloudkms.projects.locations.keyRings.cryptoKeys')