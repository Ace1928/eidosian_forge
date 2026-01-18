from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class BuildArtifacts(base.Group):
    """Manage Artifact Registry resources."""
    category = base.CI_CD_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args