from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import crawlers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions
def ValidateSchedulingFlagsForUpdate(ref, args, request, crawler):
    del ref
    return _ValidateSchedulingFlags(args, request, crawler, for_update=True)