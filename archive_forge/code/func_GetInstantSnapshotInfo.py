from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.exceptions import Error
import six
def GetInstantSnapshotInfo(ips_ref, client, messages):
    """Gets the zonal or regional instant snapshot api info.

  Args:
    ips_ref: the instant snapshot resource reference that is parsed from
      resource arguments.
    client: the compute api_tools_client.
    messages: the compute message module.

  Returns:
    _ZoneInstantSnapshot or _RegionInstantSnapshot.
  """
    if IsZonal(ips_ref):
        return _InstantSnapshot(client, ips_ref, messages)
    else:
        return _RegionInstantSnapshot(client, ips_ref, messages)