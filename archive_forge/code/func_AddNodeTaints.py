from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddNodeTaints(parser):
    parser.add_argument('--node-taints', type=arg_parsers.ArgDict(min_length=1, value_type=_ValidateNodeTaint), metavar='NODE_TAINT', help='Taints assigned to nodes of the node pool. {} {}'.format(_TAINT_FORMAT_HELP, _TAINT_EFFECT_HELP))