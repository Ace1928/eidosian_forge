from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core import properties
def AddProvisionedIopsFlag(parser, arg_parsers):
    return parser.add_argument('--provisioned-iops', type=arg_parsers.BoundedInt(), help='Provisioned IOPS of disk to create. Only for use with disks of type pd-extreme and hyperdisk-extreme.')