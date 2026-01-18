from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.runtime_config import transforms
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class RuntimeConfig(base.Group):
    """Manage runtime configuration resources."""
    category = base.MANAGEMENT_TOOLS_CATEGORY

    @staticmethod
    def Args(parser):
        parser.display_info.AddTransforms(transforms.GetTransforms())

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args
        base.DisableUserProjectQuota()