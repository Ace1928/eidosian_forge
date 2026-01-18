from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.command_lib.logs import stream
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import jobs_prep
from googlecloudsdk.command_lib.ml_engine import log_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def GetStreamLogs(asyncronous, stream_logs):
    """Return, based on the command line arguments, whether we should stream logs.

  Both arguments cannot be set (they're mutually exclusive flags) and the
  default is False.

  Args:
    asyncronous: bool, the value of the --async flag.
    stream_logs: bool, the value of the --stream-logs flag.

  Returns:
    bool, whether to stream the logs

  Raises:
    ValueError: if both asyncronous and stream_logs are True.
  """
    if asyncronous and stream_logs:
        raise ValueError('--async and --stream-logs cannot both be set.')
    if asyncronous:
        log.warning('The --async flag is deprecated, as the default behavior is to submit the job asynchronously; it can be omitted. For synchronous behavior, please pass --stream-logs.\n')
    return stream_logs