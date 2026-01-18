from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
def ApplyCdnPolicyArgs(client, args, backend_bucket, is_update=False, cleared_fields=None):
    """Applies the CdnPolicy arguments to the specified backend bucket.

  If there are no arguments related to CdnPolicy, the backend bucket remains
  unmodified.

  Args:
    client: The client used by gcloud.
    args: The arguments passed to the gcloud command.
    backend_bucket: The backend bucket object.
    is_update: True if this is called on behalf of an update command instead of
      a create command, False otherwise.
    cleared_fields: Reference to list with fields that should be cleared. Valid
      only for update command.
  """
    if backend_bucket.cdnPolicy is not None:
        cdn_policy = encoding.CopyProtoMessage(backend_bucket.cdnPolicy)
    else:
        cdn_policy = client.messages.BackendBucketCdnPolicy()
    if args.IsSpecified('signed_url_cache_max_age'):
        cdn_policy.signedUrlCacheMaxAgeSec = args.signed_url_cache_max_age
    if args.request_coalescing is not None:
        cdn_policy.requestCoalescing = args.request_coalescing
    if args.cache_mode:
        cdn_policy.cacheMode = client.messages.BackendBucketCdnPolicy.CacheModeValueValuesEnum(args.cache_mode)
    if args.client_ttl is not None:
        cdn_policy.clientTtl = args.client_ttl
    if args.default_ttl is not None:
        cdn_policy.defaultTtl = args.default_ttl
    if args.max_ttl is not None:
        cdn_policy.maxTtl = args.max_ttl
    if is_update:
        should_clean_client_ttl = args.cache_mode == 'USE_ORIGIN_HEADERS' and args.client_ttl is None
        if args.no_client_ttl or should_clean_client_ttl:
            cleared_fields.append('cdnPolicy.clientTtl')
            cdn_policy.clientTtl = None
        should_clean_default_ttl = args.cache_mode == 'USE_ORIGIN_HEADERS' and args.default_ttl is None
        if args.no_default_ttl or should_clean_default_ttl:
            cleared_fields.append('cdnPolicy.defaultTtl')
            cdn_policy.defaultTtl = None
        should_clean_max_ttl = (args.cache_mode == 'USE_ORIGIN_HEADERS' or args.cache_mode == 'FORCE_CACHE_ALL') and args.max_ttl is None
        if args.no_max_ttl or should_clean_max_ttl:
            cleared_fields.append('cdnPolicy.maxTtl')
            cdn_policy.maxTtl = None
    if args.negative_caching is not None:
        cdn_policy.negativeCaching = args.negative_caching
    negative_caching_policy = GetNegativeCachingPolicy(client, args, backend_bucket)
    if negative_caching_policy is not None:
        cdn_policy.negativeCachingPolicy = negative_caching_policy
    if args.negative_caching_policy:
        cdn_policy.negativeCaching = True
    if is_update:
        if args.no_negative_caching_policies or (args.negative_caching is not None and (not args.negative_caching)):
            cleared_fields.append('cdnPolicy.negativeCachingPolicy')
            cdn_policy.negativeCachingPolicy = []
    if args.serve_while_stale is not None:
        cdn_policy.serveWhileStale = args.serve_while_stale
    bypass_cache_on_request_headers = GetBypassCacheOnRequestHeaders(client, args)
    if bypass_cache_on_request_headers is not None:
        cdn_policy.bypassCacheOnRequestHeaders = bypass_cache_on_request_headers
    if is_update:
        if args.no_serve_while_stale:
            cleared_fields.append('cdnPolicy.serveWhileStale')
            cdn_policy.serveWhileStale = None
        if args.no_bypass_cache_on_request_headers:
            cleared_fields.append('cdnPolicy.bypassCacheOnRequestHeaders')
            cdn_policy.bypassCacheOnRequestHeaders = []
    if HasCacheKeyPolicyArgs(args):
        cdn_policy.cacheKeyPolicy = GetCacheKeyPolicy(client, args, backend_bucket)
    if cdn_policy != client.messages.BackendBucketCdnPolicy():
        backend_bucket.cdnPolicy = cdn_policy