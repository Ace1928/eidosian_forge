from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.network_connectivity import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddVPCNetworkFlag(parser):
    """Adds the --vpc-network argument to the given parser."""
    parser.add_argument('--vpc-network', required=True, help='VPC network that the spoke provides connectivity to.\n      The resource must already exist.')