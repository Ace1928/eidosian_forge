from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddEndpointSubnetId(parser):
    parser.add_argument('--endpoint-subnet-id', help='ARM ID of the subnet where the control plane load balancer is deployed. When unspecified, it defaults to the control plane subnet ID.')