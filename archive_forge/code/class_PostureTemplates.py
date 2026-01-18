from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class PostureTemplates(base.Group):
    """Manage Cloud Security Command Center (SCC) posture templates."""
    category = base.SECURITY_CATEGORY