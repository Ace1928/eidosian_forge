from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.backend_buckets import (
from googlecloudsdk.command_lib.compute.backend_services import (
from googlecloudsdk.command_lib.compute.url_maps import flags
from googlecloudsdk.command_lib.compute.url_maps import url_maps_utils
from googlecloudsdk.core import properties
import six
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class AddPathMatcher(base.UpdateCommand):
    """Add a path matcher to a URL map."""
    detailed_help = _DetailedHelp()
    BACKEND_SERVICE_ARG = None
    BACKEND_BUCKET_ARG = None
    URL_MAP_ARG = None

    @classmethod
    def Args(cls, parser):
        cls.BACKEND_BUCKET_ARG = backend_bucket_flags.BackendBucketArgumentForUrlMap()
        cls.BACKEND_SERVICE_ARG = backend_service_flags.BackendServiceArgumentForUrlMap()
        cls.URL_MAP_ARG = flags.UrlMapArgument()
        cls.URL_MAP_ARG.AddArgument(parser)
        _Args(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        return _Run(args, holder, self.URL_MAP_ARG, self.BACKEND_SERVICE_ARG, self.BACKEND_BUCKET_ARG)