from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddInstanceType(parser, kind='control plane'):
    """Adds the --instance-type flag."""
    parser.add_argument('--instance-type', help="AWS EC2 instance type for the {}'s nodes.".format(kind))