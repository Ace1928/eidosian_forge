from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class DataprocGdc(base.Group):
    """Create and manage Dataproc on GDC instances."""
    category = base.DATA_ANALYTICS_CATEGORY