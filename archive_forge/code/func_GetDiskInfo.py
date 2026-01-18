from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.core.exceptions import Error
def GetDiskInfo(disk_ref, client, messages):
    """Gets the zonal or regional disk api info.

  Args:
    disk_ref: the disk resource reference that is parsed from resource
      arguments.
    client: the compute api_tools_client.
    messages: the compute message module.

  Returns:
    _ZonalDisk or _RegionalDisk.
  """
    if IsZonal(disk_ref):
        return _ZonalDisk(client, disk_ref, messages)
    else:
        return _RegionalDisk(client, disk_ref, messages)