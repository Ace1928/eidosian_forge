from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
class AlphaNetappAdapter(BetaNetappAdapter):
    """Adapter for the Alpha Cloud NetApp Files API."""

    def __init__(self):
        super(AlphaNetappAdapter, self).__init__()
        self.release_track = base.ReleaseTrack.ALPHA
        self.client = util.GetClientInstance(release_track=self.release_track)
        self.messages = util.GetMessagesModule(release_track=self.release_track)