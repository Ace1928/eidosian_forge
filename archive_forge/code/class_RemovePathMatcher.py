from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.url_maps import flags
from googlecloudsdk.command_lib.compute.url_maps import url_maps_utils
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class RemovePathMatcher(base.UpdateCommand):
    """Remove a path matcher from a URL map."""
    detailed_help = _DetailedHelp()
    URL_MAP_ARG = None

    @classmethod
    def Args(cls, parser):
        cls.URL_MAP_ARG = flags.UrlMapArgument()
        cls.URL_MAP_ARG.AddArgument(parser)
        parser.add_argument('--path-matcher-name', required=True, help='The name of the path matcher to remove.')

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        return _Run(args, holder, self.URL_MAP_ARG)