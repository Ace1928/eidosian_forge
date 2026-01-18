from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class NetworkPeeringRoutes(base.Group):
    """Manage VMware Engine VPC peering routes using Google Cloud VMware Engine."""
    category = base.COMPUTE_CATEGORY