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
def AddCreateVersionDestroyTTL(parser, positional=False, **kwargs):
    """Add flags for specifying version destroy ttl on secret creates."""
    parser.add_argument(_ArgOrFlag('version-destroy-ttl', positional), metavar='VERSION-DESTROY-TTL', type=arg_parsers.Duration(), help='Secret Version Time To Live (TTL) after destruction request. For secret with TTL>0, version destruction does not happen immediately on calling destroy; instead, the version goes to a disabled state and destruction happens after the TTL expires. See `$ gcloud topic datetimes` for information on duration formats.', **kwargs)