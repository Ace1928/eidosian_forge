from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class HttpFilters(base.Group):
    """Manage Network Services HttpFilters."""
    category = base.MANAGEMENT_TOOLS_CATEGORY