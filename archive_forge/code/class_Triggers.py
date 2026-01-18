from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import transforms
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Triggers(base.Group):
    """Create and manage build triggers for Google Cloud Build."""
    category = base.CI_CD_CATEGORY

    @staticmethod
    def Args(parser):
        parser.display_info.AddTransforms(transforms.GetTransforms())