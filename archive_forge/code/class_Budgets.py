from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Budgets(base.Group):
    """Manage the budgets of your billing accounts."""

    def Filter(self, context, args):
        del context, args
        base.EnableUserProjectQuota()