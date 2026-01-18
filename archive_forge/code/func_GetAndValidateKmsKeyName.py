from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.filestore import filestore_client
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.filestore import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def GetAndValidateKmsKeyName(args):
    """Parse the KMS key resource arg, make sure the key format is correct."""
    kms_ref = args.CONCEPTS.kms_key.Parse()
    if kms_ref:
        return kms_ref.RelativeName()
    for keyword in ['kms-key', 'kms-keyring', 'kms-location', 'kms-project']:
        if getattr(args, keyword.replace('-', '_'), None):
            raise exceptions.InvalidArgumentException('--kms-project --kms-location --kms-keyring --kms-key', 'Specify fully qualified KMS key ID with --kms-key, or use combination of --kms-project, --kms-location, --kms-keyring and --kms-key to specify the key ID in pieces.')
    return None