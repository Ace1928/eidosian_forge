from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.url_maps import flags
from googlecloudsdk.command_lib.compute.url_maps import url_maps_utils
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class AddHostRule(base.UpdateCommand):
    """Add a rule to a URL map to map hosts to a path matcher."""
    detailed_help = _DetailedHelp()
    URL_MAP_ARG = None

    @classmethod
    def Args(cls, parser):
        cls.URL_MAP_ARG = flags.UrlMapArgument()
        cls.URL_MAP_ARG.AddArgument(parser)
        _Args(parser)

    def Run(self, args):
        """Issues requests necessary to add host rule to the Url Map."""
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        return _Run(args, holder, self.URL_MAP_ARG)