from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class ServiceLbPolicies(base.Group):
    """Manage Network Services ServiceLbPolicies."""
    category = base.MANAGEMENT_TOOLS_CATEGORY