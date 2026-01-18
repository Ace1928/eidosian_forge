from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class GrpcRoutes(base.Group):
    """Manage Network Services GrpcRoutes."""
    category = base.MANAGEMENT_TOOLS_CATEGORY