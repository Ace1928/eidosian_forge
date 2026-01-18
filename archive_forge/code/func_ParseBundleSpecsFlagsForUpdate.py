from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import crawlers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions
def ParseBundleSpecsFlagsForUpdate(ref, args, request, crawler):
    """Python hook that parses the bundle spec args into the update request.

  Args:
    ref: The crawler resource reference.
    args: The parsed args namespace.
    request: The update crawler request.
    crawler: CachedResult, The cached crawler result.
  Returns:
    Request with bundling specs set appropriately.
  """
    del ref
    if not _IsChangeBundleSpecsSpecified(args):
        return request
    bundle_specs = crawler.Get().config.bundleSpecs or []
    if args.IsSpecified('clear_bundle_specs'):
        bundle_specs = []
    if args.IsSpecified('remove_bundle_specs'):
        to_remove = set(args.remove_bundle_specs)
        bundle_specs = [b for b in bundle_specs if b not in to_remove]
    if args.IsSpecified('add_bundle_specs'):
        bundle_specs += args.add_bundle_specs
    arg_utils.SetFieldInMessage(request, 'googleCloudDatacatalogV1alpha3Crawler.config.bundleSpecs', bundle_specs)
    return request