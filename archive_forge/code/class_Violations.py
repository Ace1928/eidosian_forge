from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.base import ReleaseTrack
@base.ReleaseTracks(ReleaseTrack.GA, ReleaseTrack.BETA, ReleaseTrack.ALPHA)
class Violations(base.Group):
    """Read and list Assured Workloads Violations."""
    category = base.SECURITY_CATEGORY
    detailed_help = {'DESCRIPTION': '\n        Read and list Assured Workloads Violations.\n    '}