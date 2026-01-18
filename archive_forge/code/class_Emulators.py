from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Emulators(base.Group):
    """Set up your local development environment using emulators."""
    category = base.SDK_TOOLS_CATEGORY

    def Filter(self, context, args):
        del context, args
        base.DisableUserProjectQuota()