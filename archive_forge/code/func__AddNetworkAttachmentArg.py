from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import googlecloudsdk
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def _AddNetworkAttachmentArg(parser, required=False):
    """Adds an argument for the trigger's destination service account."""
    parser.add_argument('--network-attachment', required=required, help='The network attachment associated with the trigger that allows access to the destination VPC.')