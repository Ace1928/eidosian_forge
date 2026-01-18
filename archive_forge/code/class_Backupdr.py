from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class Backupdr(base.Group):
    """Manage Backup and DR resources."""
    category = base.STORAGE_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args