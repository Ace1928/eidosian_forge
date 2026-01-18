from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.base import ReleaseTrack
@base.ReleaseTracks(ReleaseTrack.GA, ReleaseTrack.BETA, ReleaseTrack.ALPHA)
class Workloads(base.Group):
    """Read and manipulate Assured Workloads resources."""
    category = base.SECURITY_CATEGORY
    detailed_help = {'DESCRIPTION': '\n        Create, read, update, list and delete Assured Workloads resources.\n    '}