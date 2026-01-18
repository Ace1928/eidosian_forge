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
def AddUpdateVersionDestroyTTL(parser, positional=False, **kwargs):
    """Add flags for specifying version destroy ttl on secret updates."""
    group = parser.add_group(mutex=True, help='Version destroy ttl.')
    group.add_argument(_ArgOrFlag('version-destroy-ttl', positional), metavar='VERSION-DESTROY-TTL', type=arg_parsers.Duration(), help='Secret Version TTL after destruction request. For secret with TTL>0, version destruction does not happen immediately on calling destroy; instead, the version goes to a disabled state and destruction happens after the TTL expires. See `$ gcloud topic datetimes` for information on duration formats.', **kwargs)
    group.add_argument(_ArgOrFlag('remove-version-destroy-ttl', False), action='store_true', help='If set, removes the version destroy TTL from the secret.', **kwargs)