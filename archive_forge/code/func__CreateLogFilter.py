from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.functions.v1 import util as util_v1
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.api_lib.functions.v2 import exceptions
from googlecloudsdk.api_lib.logging import common as logging_common
from googlecloudsdk.api_lib.logging import util as logging_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.functions import flags
from googlecloudsdk.command_lib.functions import util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def _CreateLogFilter(args):
    """Creates the filter for retrieving function logs based on the given args.


  Args:
    args: The arguments that were provided to this command invocation.

  Returns:
  """
    function_ref = _GetFunctionRef(args.name)
    region = properties.VALUES.functions.region.GetOrFail()
    if flags.ShouldUseGen1():
        log_filter = [_CreateGen1LogFilterBase(function_ref, region)]
    elif flags.ShouldUseGen2():
        log_filter = [_CreateGen2LogFilterBase(function_ref, region)]
    else:
        log_filter = ['({}) OR ({})'.format(_CreateGen1LogFilterBase(function_ref, region), _CreateGen2LogFilterBase(function_ref, region))]
    if args.execution_id:
        log_filter.append('labels.execution_id="{}"'.format(args.execution_id))
    if args.min_log_level:
        log_filter.append('severity>={}'.format(args.min_log_level.upper()))
    if args.end_time:
        log_filter.append('timestamp<="{}"'.format(logging_util.FormatTimestamp(args.end_time)))
    log_filter.append('timestamp>="{}"'.format(logging_util.FormatTimestamp(args.start_time or datetime.datetime.utcnow() - datetime.timedelta(days=7))))
    return ' '.join(log_filter)