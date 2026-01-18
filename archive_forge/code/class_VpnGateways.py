from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class VpnGateways(base.Group):
    """read and manipulate Highly Available VPN Gateways."""
    detailed_help = None
    category = base.NETWORKING_CATEGORY