from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
def GetLocalSsdFlag(custom_name=None):
    """Gets the --local-ssd flag."""
    help_text = '  Manage the size and the interface of local SSD to use. See\n  https://cloud.google.com/compute/docs/disks/local-ssd for more information.\n  *interface*::: The kind of disk interface exposed to the VM for this SSD. Valid\n  values are `scsi` and `nvme`. SCSI is the default and is supported by more\n  guest operating systems. NVME may provide higher performance.\n  *size*::: The size of the local SSD in base-2 GB.\n  '
    return base.Argument(custom_name if custom_name else '--local-ssd', type=arg_parsers.ArgDict(spec={'interface': lambda x: x.upper(), 'size': int}), action='append', help=help_text)