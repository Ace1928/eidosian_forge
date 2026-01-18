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
def AddCreateReplicationPolicyGroup(parser):
    """Add flags for specifying replication policy on secret creation."""
    group = parser.add_group(mutex=True, help='Replication policy.')
    group.add_argument(_ArgOrFlag('replication-policy-file', False), metavar='REPLICATION-POLICY-FILE', help='JSON or YAML file to use to read the replication policy. The file must conform to https://cloud.google.com/secret-manager/docs/reference/rest/v1/projects.secrets#replication.Set this to "-" to read from stdin.')
    subgroup = group.add_group(help='Inline replication arguments.')
    subgroup.add_argument(_ArgOrFlag('replication-policy', False), metavar='POLICY', help='The type of replication policy to apply to this secret. Allowed values are "automatic" and "user-managed". If user-managed then --locations must also be provided.')
    subgroup.add_argument(_ArgOrFlag('kms-key-name', False), metavar='KMS-KEY-NAME', help='Global KMS key with which to encrypt and decrypt the secret. Only valid for secrets with an automatic replication policy.')
    subgroup.add_argument(_ArgOrFlag('locations', False), action=arg_parsers.UpdateAction, metavar='LOCATION', type=arg_parsers.ArgList(), help='Comma-separated list of locations in which the secret should be replicated.')