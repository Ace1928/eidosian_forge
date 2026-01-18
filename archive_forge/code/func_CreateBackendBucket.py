from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import cdn_flags_utils as cdn_flags
from googlecloudsdk.command_lib.compute import signed_url_flags
from googlecloudsdk.command_lib.compute.backend_buckets import backend_buckets_utils
from googlecloudsdk.command_lib.compute.backend_buckets import flags as backend_buckets_flags
def CreateBackendBucket(self, args):
    """Creates and returns the backend bucket."""
    holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
    client = holder.client
    backend_buckets_ref = self.BACKEND_BUCKET_ARG.ResolveAsResource(args, holder.resources)
    enable_cdn = args.enable_cdn or False
    backend_bucket = client.messages.BackendBucket(description=args.description, name=backend_buckets_ref.Name(), bucketName=args.gcs_bucket_name, enableCdn=enable_cdn)
    backend_buckets_utils.ApplyCdnPolicyArgs(client, args, backend_bucket)
    if args.custom_response_header is not None:
        backend_bucket.customResponseHeaders = args.custom_response_header
    if backend_bucket.cdnPolicy is not None and backend_bucket.cdnPolicy.cacheMode and (args.enable_cdn is not False):
        backend_bucket.enableCdn = True
    if args.compression_mode is not None:
        backend_bucket.compressionMode = client.messages.BackendBucket.CompressionModeValueValuesEnum(args.compression_mode)
    return backend_bucket