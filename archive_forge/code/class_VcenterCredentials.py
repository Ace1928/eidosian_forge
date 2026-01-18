from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class VcenterCredentials(base.Group):
    """Manage VMware vCenter credentials using Google Cloud VMware Engine."""
    category = base.COMPUTE_CATEGORY