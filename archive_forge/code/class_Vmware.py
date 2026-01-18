from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Vmware(base.Group):
    """Manage Google Cloud VMware Engine resources."""
    category = base.COMPUTE_CATEGORY