from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class FutureReservations(base.Group):
    """Manage Compute Engine future reservations."""
    category = base.COMPUTE_CATEGORY
    detailed_help = {'DESCRIPTION': '\n        Manage Compute Engine future reservations.\n    '}