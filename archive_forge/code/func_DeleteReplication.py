from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def DeleteReplication(self, replication_ref, async_):
    """Delete an existing Cloud NetApp Volume Replication."""
    request = self.messages.NetappProjectsLocationsVolumesReplicationsDeleteRequest(name=replication_ref.RelativeName())
    return self._DeleteReplication(async_, request)