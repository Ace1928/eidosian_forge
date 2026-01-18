from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Dataproc(base.Group):
    """Create and manage Google Cloud Dataproc clusters and jobs."""
    category = base.DATA_ANALYTICS_CATEGORY
    detailed_help = DETAILED_HELP

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args
        base.DisableUserProjectQuota()
        self.EnableSelfSignedJwtForTracks([base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA])