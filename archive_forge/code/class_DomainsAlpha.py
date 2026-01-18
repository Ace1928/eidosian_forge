from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class DomainsAlpha(base.Group):
    """Manage domains for your Google Cloud projects."""
    category = base.NETWORKING_CATEGORY