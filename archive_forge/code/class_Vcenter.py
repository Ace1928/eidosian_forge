from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Vcenter(base.Group):
    """Manage vCenter resources in Google Cloud VMware Engine."""
    category = base.COMPUTE_CATEGORY