from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import crawlers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions
def ValidateScopeFlagsForCreate(ref, args, request):
    """Validates scope flags for create.

  Args:
    ref: The crawler resource reference.
    args: The parsed args namespace.
    request: The create request.
  Returns:
    The request, if the crawl scope configuration is valid.
  Raises:
    InvalidCrawlScopeError: If the crawl scope configuration is not valid.
  """
    del ref
    if args.IsSpecified('buckets') and args.crawl_scope != 'bucket':
        raise InvalidCrawlScopeError('Argument `--buckets` is only valid for bucket-scoped crawlers. Use `--crawl-scope=bucket` to specify a bucket-scoped crawler.')
    if not args.IsSpecified('buckets') and args.crawl_scope == 'bucket':
        raise InvalidCrawlScopeError('Argument `--buckets` must be provided when creating a bucket-scoped crawler.')
    return request