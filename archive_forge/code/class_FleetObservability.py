from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base as calliope_base
@calliope_base.ReleaseTracks(calliope_base.ReleaseTrack.ALPHA, calliope_base.ReleaseTrack.BETA, calliope_base.ReleaseTrack.GA)
class FleetObservability(calliope_base.Group):
    """Manage Fleet Observability Feature."""
    category = calliope_base.COMPUTE_CATEGORY