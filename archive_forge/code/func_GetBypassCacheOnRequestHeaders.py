from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
def GetBypassCacheOnRequestHeaders(client, args):
    """Returns bypass cache on request headers.

  Args:
    client: The client used by gcloud.
    args: The arguments passed to the gcloud command.

  Returns:
    The bypass cache on request headers.
  """
    bypass_cache_on_request_headers = None
    if args.bypass_cache_on_request_headers:
        bypass_cache_on_request_headers = []
        for header in args.bypass_cache_on_request_headers:
            bypass_cache_on_request_headers.append(client.messages.BackendBucketCdnPolicyBypassCacheOnRequestHeader(headerName=header))
    return bypass_cache_on_request_headers