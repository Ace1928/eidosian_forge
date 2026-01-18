from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class ApphubAlpha(base.Group):
    """Manage App Hub resources."""
    category = base.MANAGEMENT_TOOLS_CATEGORY