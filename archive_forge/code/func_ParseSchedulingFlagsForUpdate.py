from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import crawlers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions
def ParseSchedulingFlagsForUpdate(ref, args, request):
    del ref
    client = crawlers.CrawlersClient()
    messages = client.messages
    run_option = 'scheduled' if args.IsSpecified('run_schedule') else args.run_option
    return _SetRunOptionInRequest(run_option, args.run_schedule, request, messages)