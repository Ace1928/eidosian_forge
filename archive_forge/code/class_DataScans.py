from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class DataScans(base.Group):
    """Manage Dataplex Datascan."""
    category = base.DATA_ANALYTICS_CATEGORY