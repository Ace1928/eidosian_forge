from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class OpsagentsPolicies(base.Group):
    """Manage Operation Suite (Ops) agents policies that install, update, and uninstall agents for Compute Engine VM instances."""