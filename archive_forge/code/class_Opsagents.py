from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Opsagents(base.Group):
    """Manage Operation Suite (Ops) agents for Compute Engine VM instances."""

    def Filter(self, context, args):
        del context, args
        base.EnableUserProjectQuota()