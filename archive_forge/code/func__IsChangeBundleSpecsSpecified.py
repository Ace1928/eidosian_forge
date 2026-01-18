from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import crawlers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions
def _IsChangeBundleSpecsSpecified(args):
    return args.IsSpecified('add_bundle_specs') or args.IsSpecified('remove_bundle_specs') or args.IsSpecified('clear_bundle_specs')