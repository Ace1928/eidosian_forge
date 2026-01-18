from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class CloudQuotas(base.Group):
    """Manage Cloud Quotas quota info and quota preferences."""
    category = base.API_PLATFORM_AND_ECOSYSTEMS_CATEGORY

    def Filter(self, context, args):
        del context, args
        base.DisableUserProjectQuota()