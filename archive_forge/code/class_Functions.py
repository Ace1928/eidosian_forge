from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.api_lib.functions import transforms
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Functions(base.Group):
    """Manage Google Cloud Functions."""
    category = base.COMPUTE_CATEGORY

    @staticmethod
    def Args(parser):
        parser.display_info.AddTransforms(transforms.GetTransforms())

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args
        base.DisableUserProjectQuota()