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
def AddVersionEtag(parser):
    """Add flag for specifying the current secret version etag."""
    parser.add_argument(_ArgOrFlag('etag', False), metavar='ETAG', help="Current entity tag (ETag) of the secret version. If this flag is defined, the version is updated only if the ETag provided matched the current version's ETag.")