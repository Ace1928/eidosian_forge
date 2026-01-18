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
def _PopulateCloudTasksArgs(queue, cur_queue_state, ct_expected_args):
    """Builds placeholder command line args to pass on to Cloud Tasks API.

  Most of Cloud Tasks functions use args passed in during CLI invocation. To
  reuse those functions without extensive rework on their implementation, we
  recreate the args in the format that those functions expect.

  Args:
    queue: third_party.appengine.api.queueinfo.QueueEntry, The QueueEntry
      instance generated from the parsed YAML file.
    cur_queue_state: apis.cloudtasks.<ver>.cloudtasks_<ver>_messages.Queue,
      The Queue instance fetched from the backend if it exists, None otherwise.
    ct_expected_args: A list of expected args that we need to initialize before
      forwarding to Cloud Tasks APIs.

  Returns:
    argparse.Namespace, A placeholder args namespace built to pass on forwards
    to Cloud Tasks API.
  """
    cloud_task_args = parser_extensions.Namespace()
    for task_flag in ct_expected_args:
        setattr(cloud_task_args, task_flag, None)
    used_default_value_for_min_backoff = False
    for old_arg, new_arg in constants.APP_TO_TASKS_ATTRIBUTES_MAPPING.items():
        old_arg_list = old_arg.split('.')
        value = queue
        for old_arg_sub in old_arg_list:
            if not hasattr(value, old_arg_sub):
                value = None
                break
            value = getattr(value, old_arg_sub)
        if value or (value is not None and new_arg in ('max_attempts',)):
            if old_arg in CONVERSION_FUNCTIONS:
                value = CONVERSION_FUNCTIONS[old_arg](value)
            if not cur_queue_state or new_arg in ('name', 'type', 'min_backoff', 'max_backoff') or _DoesAttributeNeedToBeUpdated(cur_queue_state, new_arg, value):
                _SetSpecifiedArg(cloud_task_args, new_arg, value)
        else:
            if queue.mode == constants.PULL_QUEUE:
                default_values = constants.PULL_QUEUES_APP_DEPLOY_DEFAULT_VALUES
            else:
                default_values = constants.PUSH_QUEUES_APP_DEPLOY_DEFAULT_VALUES
            if new_arg in default_values:
                if new_arg == 'min_backoff':
                    used_default_value_for_min_backoff = True
                value = default_values[new_arg]
                if not cur_queue_state or new_arg in ('min_backoff', 'max_backoff') or _DoesAttributeNeedToBeUpdated(cur_queue_state, new_arg, value):
                    _SetSpecifiedArg(cloud_task_args, new_arg, value)
        setattr(cloud_task_args, new_arg, value)
    _PostProcessMinMaxBackoff(cloud_task_args, used_default_value_for_min_backoff, cur_queue_state)
    _PostProcessRoutingOverride(cloud_task_args, cur_queue_state)
    return cloud_task_args