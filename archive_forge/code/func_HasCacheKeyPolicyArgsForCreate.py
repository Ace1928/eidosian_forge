from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def HasCacheKeyPolicyArgsForCreate(args):
    """Returns true if create request requires a CacheKeyPolicy message.

  Args:
    args: The arguments passed to the gcloud command.

  Returns:
    True if there are cache key policy related arguments which require adding
    a CacheKeyPolicy message in the create request.
  """
    return not args.cache_key_include_host or not args.cache_key_include_protocol or (not args.cache_key_include_query_string) or args.IsSpecified('cache_key_query_string_whitelist') or args.IsSpecified('cache_key_query_string_blacklist') or args.IsSpecified('cache_key_include_http_header') or args.IsSpecified('cache_key_include_named_cookie')