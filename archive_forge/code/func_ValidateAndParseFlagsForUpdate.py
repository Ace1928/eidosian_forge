from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import crawlers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions
def ValidateAndParseFlagsForUpdate(ref, args, request):
    """Python hook that validates and parses crawler config flags.

  Normally all the functions called here would be provided directly as
  modify_request_hooks in the update command YAML file. However, this would
  require setting read_modify_update: True to obtain the existing crawler
  beforehand, incurring an extra GET API call that may be unnecessary depending
  on what fields need to be updated.

  Args:
    ref: The crawler resource reference.
    args: The parsed args namespace.
    request: The update crawler request.
  Returns:
    Request with scope and scheduling flags set appropriately.
  Raises:
    InvalidCrawlScopeError: If the crawl scope configuration is not valid.
    InvalidRunOptionError: If the scheduling configuration is not valid.
  """
    client = crawlers.CrawlersClient()
    crawler = repeated.CachedResult.FromFunc(client.Get, ref)
    request = ValidateScopeFlagsForUpdate(ref, args, request, crawler)
    request = ValidateSchedulingFlagsForUpdate(ref, args, request, crawler)
    request = ParseScopeFlagsForUpdate(ref, args, request, crawler)
    request = ParseSchedulingFlagsForUpdate(ref, args, request)
    request = ParseBundleSpecsFlagsForUpdate(ref, args, request, crawler)
    return request