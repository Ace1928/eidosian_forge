from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class TelcoAutomationAlpha(base.Group):
    """Manage Telco Automation resources."""
    category = base.NETWORKING_CATEGORY