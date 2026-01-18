from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core import properties
def AddProvisionedThroughputFlag(parser, arg_parsers):
    return parser.add_argument('--provisioned-throughput', type=arg_parsers.BoundedInt(), help='Provisioned throughput of disk to create. The throughput unit is  MB per sec.  Only for use with disks of type hyperdisk-throughput.')