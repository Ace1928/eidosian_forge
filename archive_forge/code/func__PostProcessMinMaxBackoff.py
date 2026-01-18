from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.tasks import task_queues_convertors as convertors
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import urllib
def _PostProcessMinMaxBackoff(cloud_task_args, used_default_value_for_min_backoff, cur_queue_state):
    """Checks min and max backoff values and updates the other value if needed.

  When uploading via queue.yaml files, if only one of the backoff values is
  specified, the other value will automatically be updated to the default
  value. If the default value does not satisfy the condition
  min_backoff <= max_backoff, then it is set equal to the other backoff value.

  Args:
    cloud_task_args: argparse.Namespace, A placeholder args namespace built to
      pass on forwards to Cloud Tasks API.
    used_default_value_for_min_backoff: A boolean value telling us if we used
      a default value for min_backoff or if it was specified explicitly in the
      YAML file.
    cur_queue_state: apis.cloudtasks.<ver>.cloudtasks_<ver>_messages.Queue,
      The Queue instance fetched from the backend if it exists, None otherwise.
  """
    if cloud_task_args.type == 'pull':
        return
    min_backoff = convertors.CheckAndConvertStringToFloatIfApplicable(cloud_task_args.min_backoff)
    max_backoff = convertors.CheckAndConvertStringToFloatIfApplicable(cloud_task_args.max_backoff)
    if min_backoff > max_backoff:
        if used_default_value_for_min_backoff:
            min_backoff = max_backoff
            cloud_task_args.min_backoff = cloud_task_args.max_backoff
            _SetSpecifiedArg(cloud_task_args, 'min_backoff', cloud_task_args.max_backoff)
        else:
            max_backoff = min_backoff
            cloud_task_args.max_backoff = cloud_task_args.min_backoff
            _SetSpecifiedArg(cloud_task_args, 'max_backoff', cloud_task_args.min_backoff)
    if cur_queue_state and cur_queue_state.retryConfig:
        old_min_backoff = convertors.CheckAndConvertStringToFloatIfApplicable(cur_queue_state.retryConfig.minBackoff)
        old_max_backoff = convertors.CheckAndConvertStringToFloatIfApplicable(cur_queue_state.retryConfig.maxBackoff)
        if max_backoff == old_max_backoff and min_backoff == old_min_backoff:
            _DeleteSpecifiedArg(cloud_task_args, 'min_backoff')
            cloud_task_args.min_backoff = None
            _DeleteSpecifiedArg(cloud_task_args, 'max_backoff')
            cloud_task_args.max_backoff = None