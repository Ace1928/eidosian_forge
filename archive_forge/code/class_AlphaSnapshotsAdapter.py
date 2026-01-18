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
class AlphaSnapshotsAdapter(BetaSnapshotsAdapter):
    """Adapter for the Alpha Cloud NetApp Files API Snapshot resource."""

    def __init__(self):
        super(AlphaSnapshotsAdapter, self).__init__()
        self.release_track = base.ReleaseTrack.ALPHA
        self.client = netapp_api_util.GetClientInstance(release_track=self.release_track)
        self.messages = netapp_api_util.GetMessagesModule(release_track=self.release_track)