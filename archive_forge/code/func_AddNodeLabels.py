from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddNodeLabels(parser):
    """Adds the --node-labels flag."""
    parser.add_argument('--node-labels', type=arg_parsers.ArgDict(min_length=1), metavar='NODE_LABEL', help="Labels assigned to the node pool's nodes.")