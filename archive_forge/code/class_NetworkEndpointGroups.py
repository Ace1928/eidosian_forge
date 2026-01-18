from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class NetworkEndpointGroups(base.Group):
    """Read and manipulate Compute Engine network endpoint groups."""
    category = base.NETWORKING_CATEGORY