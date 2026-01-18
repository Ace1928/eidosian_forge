from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.base import ReleaseTrack
@base.ReleaseTracks(ReleaseTrack.GA, ReleaseTrack.BETA, ReleaseTrack.ALPHA)
class Assured(base.Group):
    """Read and manipulate Assured Workloads data controls."""
    category = base.SECURITY_CATEGORY
    detailed_help = {'DESCRIPTION': '\n        Read and manipulate Assured Workloads data controls.\n    '}

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args