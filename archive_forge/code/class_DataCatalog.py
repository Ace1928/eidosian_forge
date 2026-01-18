from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class DataCatalog(base.Group):
    """Manage Data Catalog resources."""
    category = base.DATA_ANALYTICS_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args