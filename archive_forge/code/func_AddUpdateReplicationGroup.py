from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.command_lib.secrets import completers as secrets_completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import resources
def AddUpdateReplicationGroup(parser):
    """Add flags for specifying replication policy updates."""
    group = parser.add_group(mutex=True, help='Replication update.')
    group.add_argument(_ArgOrFlag('remove-cmek', False), action='store_true', help='Remove customer managed encryption key so that future versions will be encrypted by a Google managed encryption key.')
    subgroup = group.add_group(help='CMEK Update.')
    subgroup.add_argument(_ArgOrFlag('set-kms-key', False), metavar='SET-KMS-KEY', help='New KMS key with which to encrypt and decrypt future secret versions.')
    subgroup.add_argument(_ArgOrFlag('location', False), metavar='REPLICA-LOCATION', help='Location of replica to update. For secrets with automatic replication policies, this can be omitted.')