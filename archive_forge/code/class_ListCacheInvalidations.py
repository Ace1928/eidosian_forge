from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.url_maps import flags
from googlecloudsdk.command_lib.compute.url_maps import url_maps_utils
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_projector
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ListCacheInvalidations(base.ListCommand):
    """List Cloud CDN cache invalidations for a URL map."""
    detailed_help = _DetailedHelp()

    @staticmethod
    def _Flags(parser):
        parser.add_argument('--limit', type=arg_parsers.BoundedInt(1, sys.maxsize, unlimited=True), help='The maximum number of invalidations to list.')

    @classmethod
    def Args(cls, parser):
        cls.URL_MAP_ARG = flags.UrlMapArgument()
        cls.URL_MAP_ARG.AddArgument(parser, operation_type='describe')
        parser.display_info.AddFormat('        table(\n          description,\n          operation_http_status():label=HTTP_STATUS,\n          status,\n          insertTime:label=TIMESTAMP\n        )')

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        return _Run(args, holder, self.URL_MAP_ARG)