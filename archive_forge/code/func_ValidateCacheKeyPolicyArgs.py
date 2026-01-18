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
def ValidateCacheKeyPolicyArgs(cache_key_policy_args):
    include_query_string = cache_key_policy_args.cache_key_include_query_string is None or cache_key_policy_args.cache_key_include_query_string
    if not include_query_string:
        if cache_key_policy_args.cache_key_query_string_whitelist is not None or cache_key_policy_args.cache_key_query_string_blacklist is not None:
            raise CacheKeyQueryStringException()