from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class NetworkConnectivity(base.Group):
    """Manage Network Connectivity Center resources."""
    category = base.NETWORKING_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args