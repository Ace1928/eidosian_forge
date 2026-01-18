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
def DeployQueuesYamlFile(tasks_api, config, all_queues_in_db_dict, ct_api_version=base.ReleaseTrack.BETA):
    """Perform a deployment based on the parsed 'queue.yaml' file.

  Args:
    tasks_api: api_lib.tasks.<Alpha|Beta|GA>ApiAdapter, Cloud Tasks API needed
      for doing queue based operations.
    config: A yaml_parsing.ConfigYamlInfo object for the parsed YAML file we
      are going to process.
    all_queues_in_db_dict: A dictionary with queue names as keys and
      corresponding apis.cloudtasks.<ver>.cloudtasks_<ver>_messages.Queue
      objects as values
    ct_api_version: The Cloud Tasks API version we want to use.

  Returns:
    A list of responses received from the Cloud Tasks APIs representing queue
    states for every call made to modify the attributes of a queue.
  """

    class _PlaceholderQueueRef:
        """A placeholder class to simulate queue_ref resource objects used in CT APIs.

    This class simulates the behaviour of the resource object returned by
    tasks.parsers.ParseQueue(...) function. We use this placeholder class
    instead of creating an actual resource instance because otherwise it takes
    roughly 2 minutes to create resource instances for a 1000 queues.

    Attributes:
      _relative_path: A string representing the full path for a queue in the
        format: 'projects/<project>/locations/<location>/queues/<queue>'
    """

        def __init__(self, relative_path):
            """Initializes the instance and sets the relative path."""
            self._relative_path = relative_path

        def RelativeName(self):
            """Gets the string representing the full path for a queue.

      This is the only function we are currently using in CT APIs for the
      queue_ref resource object.

      Returns:
        A string representing the full path for a queue in the following
        format: 'projects/<project>/locations/<location>/queues/<queue>'
      """
            return self._relative_path
    queue_yaml = config.parsed
    resume_paused_queues = queue_yaml.resume_paused_queues != 'False'
    queues_client = tasks_api.queues
    queues_not_present_in_yaml = set(all_queues_in_db_dict.keys())
    queue_ref = parsers.ParseQueue('a')
    queue_ref_stub = queue_ref.RelativeName()[:-1]
    task_args = flags._PushQueueFlags(release_track=ct_api_version)
    task_args.append(base.Argument('--max_burst_size', type=int, help=''))
    expected_args = []
    for task_flag in task_args:
        new_arg = task_flag.args[0][2:].replace('-', '_')
        expected_args.extend((new_arg, 'clear_{}'.format(new_arg)))
    responses = []
    if queue_yaml.queue is None:
        queue_yaml.queue = []
    for queue in queue_yaml.queue:
        if queue.name in queues_not_present_in_yaml:
            queues_not_present_in_yaml.remove(queue.name)
        queue_ref = _PlaceholderQueueRef('{}{}'.format(queue_ref_stub, queue.name))
        cur_queue_object = all_queues_in_db_dict.get(queue.name, None)
        cloud_task_args = _PopulateCloudTasksArgs(queue, cur_queue_object, expected_args)
        rate_to_set = cloud_task_args.GetValue('max_dispatches_per_second')
        if resume_paused_queues and cur_queue_object and (rate_to_set or queue.mode == constants.PULL_QUEUE) and (cur_queue_object.state in (cur_queue_object.state.DISABLED, cur_queue_object.state.PAUSED)):
            queues_client.Resume(queue_ref)
        elif cur_queue_object and (not rate_to_set) and (cur_queue_object.state == cur_queue_object.state.RUNNING) and (queue.mode in (None, constants.PUSH_QUEUE)):
            queues_client.Pause(queue_ref)
        if not _AnyUpdatableFields(cloud_task_args):
            continue
        queue_config = parsers.ParseCreateOrUpdateQueueArgs(cloud_task_args, constants.PUSH_QUEUE, tasks_api.messages, release_track=ct_api_version, http_queue=False)
        updated_fields = parsers.GetSpecifiedFieldsMask(cloud_task_args, constants.PUSH_QUEUE, release_track=ct_api_version)
        if not cur_queue_object:
            updated_fields.extend(['taskTtl', 'tombstoneTtl'])
        app_engine_routing_override = queue_config.appEngineHttpQueue.appEngineRoutingOverride if queue_config.appEngineHttpQueue is not None else None
        response = queues_client.Patch(queue_ref, updated_fields, retry_config=queue_config.retryConfig, rate_limits=queue_config.rateLimits, app_engine_routing_override=app_engine_routing_override, task_ttl=constants.MAX_TASK_TTL if not cur_queue_object else None, task_tombstone_ttl=constants.MAX_TASK_TOMBSTONE_TTL if not cur_queue_object else None, queue_type=queue_config.type)
        responses.append(response)
        if not cur_queue_object and (not rate_to_set) and (queue.mode == constants.PUSH_QUEUE):
            queues_client.Pause(queue_ref)
    for queue_name in queues_not_present_in_yaml:
        if queue_name == 'default':
            continue
        queue = all_queues_in_db_dict[queue_name]
        if queue.state in (queue.state.PAUSED, queue.state.DISABLED):
            continue
        queue_ref = _PlaceholderQueueRef('{}{}'.format(queue_ref_stub, queue_name))
        queues_client.Pause(queue_ref)
    return responses