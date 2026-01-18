from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddSpotInstanceTypes(parser, kind='node pool'):
    """Adds the --spot-instance-types flag."""
    parser.add_argument('--spot-instance-types', type=arg_parsers.ArgList(), metavar='INSTANCE_TYPE', help="List of AWS EC2 instance types for creating a spot {}'s nodes. The specified instance types must have the same CPU architecture, the same number of CPUs and memory. You can use the Amazon EC2 Instance Selector tool (https://github.com/aws/amazon-ec2-instance-selector) to choose instance types with matching CPU and memory configurations.".format(kind))