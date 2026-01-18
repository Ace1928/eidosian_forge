from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class VpnConnections(base.Group):
    """Manage Edge VPN connections between an Edge Container cluster and a VPC network."""