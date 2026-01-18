from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddSubnetworkArg(parser):
    parser.add_argument('--subnetwork', type=str, help='Specifies the subnetwork this network interface belongs to.')