from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class DeploymentResourcePools(base.Group):
    """Manage Vertex AI deployment resource pools."""
    category = base.VERTEX_AI_CATEGORY