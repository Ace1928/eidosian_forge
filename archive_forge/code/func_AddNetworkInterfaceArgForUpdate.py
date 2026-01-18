from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddNetworkInterfaceArgForUpdate(parser):
    parser.add_argument('--network-interface', default='nic0', help='The name of the network interface to update.')