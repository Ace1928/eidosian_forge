from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import crawlers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions
def ParseScopeFlagsForCreate(ref, args, request):
    """Python hook that parses the crawl scope args into the create request.

  Args:
    ref: The crawler resource reference.
    args: The parsed args namespace.
    request: The create crawler request.
  Returns:
    Request with crawl scope set appropriately.
  """
    del ref
    client = crawlers.CrawlersClient()
    messages = client.messages
    if args.IsSpecified('buckets'):
        buckets = [messages.GoogleCloudDatacatalogV1alpha3BucketSpec(bucket=b) for b in args.buckets]
    else:
        buckets = None
    return _SetScopeInRequest(args.crawl_scope, buckets, request, messages)