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
def AddReplicationPolicyFile(parser, positional=False, **kwargs):
    parser.add_argument(_ArgOrFlag('replication-policy-file', positional), metavar='REPLICATION-POLICY-FILE', help='JSON or YAML file to use to read the replication policy. The file must conform to https://cloud.google.com/secret-manager/docs/reference/rest/v1/projects.secrets#replication.Set this to "-" to read from stdin.', **kwargs)