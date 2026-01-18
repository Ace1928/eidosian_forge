from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def MakeDiskTypeArg(regional):
    return compute_flags.ResourceArgument(resource_name='disk type', completer=compute_completers.DiskTypesCompleter, name='DISK_TYPE', zonal_collection='compute.diskTypes', regional_collection='compute.regionDiskTypes' if regional else None, region_hidden=not regional)