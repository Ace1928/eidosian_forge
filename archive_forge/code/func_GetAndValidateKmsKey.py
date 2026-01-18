from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ai import constants
def GetAndValidateKmsKey(args):
    """Parse CMEK resource arg, and check if the arg was partially specified."""
    if hasattr(args.CONCEPTS, 'kms_key'):
        kms_ref = args.CONCEPTS.kms_key.Parse()
        if kms_ref:
            return kms_ref.RelativeName()
        else:
            for keyword in ['kms_key', 'kms_keyring', 'kms_location', 'kms_project']:
                if getattr(args, keyword, None):
                    raise exceptions.InvalidArgumentException('--kms-key', 'Encryption key not fully specified.')