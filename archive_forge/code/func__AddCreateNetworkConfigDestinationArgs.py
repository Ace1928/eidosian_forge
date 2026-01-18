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
def _AddCreateNetworkConfigDestinationArgs(parser, required=False, hidden=False):
    """Adds arguments related to trigger's Network Config destination for create operations."""
    network_config_group = parser.add_group(required=required, hidden=hidden, help='Flags for specifying a Network Config for the destination.')
    _AddNetworkAttachmentArg(network_config_group, required=True)