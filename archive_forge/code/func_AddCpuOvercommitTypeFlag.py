from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.util import scaled_integer
import six
def AddCpuOvercommitTypeFlag(parser):
    parser.add_argument('--cpu-overcommit-type', choices=['enabled', 'none'], help='CPU overcommit type for nodes created based on this template. To overcommit CPUs on a VM, set --cpu-overcommit-type equal to either standard or none, and then when creating a VM, specify a value for the --min-node-cpu flag. Lower values for --min-node-cpu specify a higher overcommit ratio, that is, proportionally more vCPUs in relation to physical CPUs. You can only overcommit CPUs on VMs that are scheduled on nodes that support it.')