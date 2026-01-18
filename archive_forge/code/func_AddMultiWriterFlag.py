from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core import properties
def AddMultiWriterFlag(parser):
    return parser.add_argument('--multi-writer', action='store_true', help='\n      Create the disk in multi-writer mode so that it can be attached\n      with read-write access to two VMs. The multi-writer feature requires\n      specialized filesystems, among other restrictions. For more information,\n      see\n      https://cloud.google.com/compute/docs/disks/sharing-disks-between-vms.\n      ')