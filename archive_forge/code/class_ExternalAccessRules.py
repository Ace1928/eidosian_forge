from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ExternalAccessRules(base.Group):
    """Manage VMware Engine external access firewall rules in Google Cloud VMware Engine."""
    category = base.COMPUTE_CATEGORY