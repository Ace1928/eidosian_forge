from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.backend_buckets import flags as backend_bucket_flags
from googlecloudsdk.command_lib.compute.backend_services import flags as backend_service_flags
from googlecloudsdk.command_lib.compute.url_maps import flags
from googlecloudsdk.command_lib.compute.url_maps import url_maps_utils
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class SetDefaultService(base.UpdateCommand):
    """Change the default service or default bucket of a URL map."""
    detailed_help = _DetailedHelp()
    BACKEND_BUCKET_ARG = None
    BACKEND_SERVICE_ARG = None
    URL_MAP_ARG = None

    @classmethod
    def Args(cls, parser):
        cls.BACKEND_BUCKET_ARG = backend_bucket_flags.BackendBucketArgumentForUrlMap(required=False)
        cls.BACKEND_SERVICE_ARG = backend_service_flags.BackendServiceArgumentForUrlMap(required=False)
        cls.URL_MAP_ARG = flags.UrlMapArgument()
        cls.URL_MAP_ARG.AddArgument(parser)
        _Args(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        return _Run(args, holder, self.BACKEND_BUCKET_ARG, self.BACKEND_SERVICE_ARG, self.URL_MAP_ARG)