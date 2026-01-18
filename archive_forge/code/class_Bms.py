from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class Bms(base.Group):
    """Manage Bare Metal Solution resources."""
    category = base.COMPUTE_CATEGORY