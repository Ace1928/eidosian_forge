from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class PostureOperations(base.Group):
    """Manage Cloud Security Command Center (SCC) posture operations."""
    category = base.SECURITY_CATEGORY