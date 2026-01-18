from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Bq(base.Group):
    """Interact with and manage resources in Google BigQuery."""
    category = base.BIG_DATA_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args
        self.EnableSelfSignedJwtForTracks([base.ReleaseTrack.ALPHA])