from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
class BetaVolumesAdapter(VolumesAdapter):
    """Adapter for the Beta Cloud NetApp Files API Volume resource."""

    def __init__(self):
        super(BetaVolumesAdapter, self).__init__()
        self.release_track = base.ReleaseTrack.BETA
        self.client = util.GetClientInstance(release_track=self.release_track)
        self.messages = util.GetMessagesModule(release_track=self.release_track)