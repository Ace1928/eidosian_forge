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
def AddRegionalKmsKeyName(parser, positional=False, **kwargs):
    """Add flag for specifying the regional KMS key name."""
    parser.add_argument(_ArgOrFlag('regional-kms-key-name', positional), metavar='KMS-KEY-NAME', help='Regional KMS key with which to encrypt and decrypt the secret. Only valid for regional secrets.', hidden=True, **kwargs)