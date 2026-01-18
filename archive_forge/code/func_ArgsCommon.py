from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import batch_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.networks.peerings import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
@classmethod
def ArgsCommon(cls, parser):
    parser.add_argument('name', help='The name of the peering.')
    parser.add_argument('--network', required=True, help='The name of the network in the current project to be peered with the peer network.')
    parser.add_argument('--peer-network', required=True, help='The name of the network to be peered with the current network.')
    parser.add_argument('--peer-project', required=False, help='The name of the project for the peer network.  If not specified, defaults to current project.')
    base.ASYNC_FLAG.AddToParser(parser)
    flags.AddImportCustomRoutesFlag(parser)
    flags.AddExportCustomRoutesFlag(parser)
    flags.AddImportSubnetRoutesWithPublicIpFlag(parser)
    flags.AddExportSubnetRoutesWithPublicIpFlag(parser)
    flags.AddStackType(parser)