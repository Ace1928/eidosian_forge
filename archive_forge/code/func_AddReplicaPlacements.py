from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddReplicaPlacements(parser):
    parser.add_argument('--replica-placements', type=arg_parsers.ArgList(element_type=_ReplicaPlacementStrToObject), metavar='REPLICA_PLACEMENT', help='Placement info for the control plane replicas. {}'.format(_REPLICAPLACEMENT_FORMAT_HELP))