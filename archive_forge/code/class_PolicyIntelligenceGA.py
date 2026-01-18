from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class PolicyIntelligenceGA(base.Group):
    """A platform to help better understand, use and manage policies at scale."""
    category = base.IDENTITY_AND_SECURITY_CATEGORY

    def Filter(self, context, args):
        """Enables User-Project override for this surface."""
        base.EnableUserProjectQuota()
        base.RequireProjectID(args)