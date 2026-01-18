from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import crawlers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions
def _ValidateSchedulingFlags(args, request, crawler=None, for_update=False):
    """Validates scheduling flags.

  Args:
    args: The parsed args namespace.
    request: The create or update request.
    crawler: CachedResult, The cached crawler result.
    for_update: If the request is for update instead of create.
  Returns:
    The request, if the scheduling configuration is valid.
  Raises:
    InvalidRunOptionError: If the scheduling configuration is not valid.
  """
    if args.run_option == 'scheduled' and (not args.IsSpecified('run_schedule')):
        raise InvalidRunOptionError('Argument `--run-schedule` must be provided if `--run-option=scheduled` was specified.')
    if args.run_option != 'scheduled' and args.IsSpecified('run_schedule'):
        if not for_update or args.IsSpecified('run_option') or crawler.Get().config.scheduledRun is None:
            raise InvalidRunOptionError('Argument `--run-schedule` can only be provided for scheduled crawlers. Use `--run-option=scheduled` to specify a scheduled crawler.')
    return request