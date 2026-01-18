from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.projects import util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Mps(base.Group):
    """Manage Marketplace Solutions resources."""
    category = base.COMPUTE_CATEGORY

    @staticmethod
    def Args(parser):
        parser.display_info.AddUriFunc(util.ProjectsUriFunc)

    def Filter(self, context, args):
        del context, args
        base.DisableUserProjectQuota()