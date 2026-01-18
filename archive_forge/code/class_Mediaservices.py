from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Mediaservices(base.Group):
    """Manage Cloud Media Services."""
    category = base.SOLUTIONS_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args