from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Serverless(base.Group):
    """Manage your Cloud Run resources."""
    category = base.COMPUTE_CATEGORY
    detailed_help = DETAILED_HELP

    def Filter(self, context, args):
        """Runs before any commands in this group."""
        base.RequireProjectID(args)
        del context, args