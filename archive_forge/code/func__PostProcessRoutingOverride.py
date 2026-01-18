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
def _PostProcessRoutingOverride(cloud_task_args, cur_queue_state):
    """Checks if service and target values need to be updated for host URL.

  An app engine host URL may have optionally version_dot_service appended to
  the URL if specified via 'routing_override'. Here we check the existing URL
  and make sure the service & target values are only updated when need be.

  Args:
    cloud_task_args: argparse.Namespace, A placeholder args namespace built to
      pass on forwards to Cloud Tasks API.
    cur_queue_state: apis.cloudtasks.<ver>.cloudtasks_<ver>_messages.Queue,
      The Queue instance fetched from the backend if it exists, None otherwise.
  """
    try:
        host_url = cur_queue_state.appEngineHttpQueue.appEngineRoutingOverride.host
    except AttributeError:
        return
    if cloud_task_args.IsSpecified('routing_override'):
        targets = []
        if 'version' in cloud_task_args.routing_override:
            targets.append(cloud_task_args.routing_override['version'])
        if 'service' in cloud_task_args.routing_override:
            targets.append(cloud_task_args.routing_override['service'])
        targets_sub_url = '.'.join(targets)
        targets_sub_url_and_project = '{}.{}.'.format(targets_sub_url, properties.VALUES.core.project.Get())
        if host_url.startswith(targets_sub_url_and_project):
            del cloud_task_args._specified_args['routing_override']
            cloud_task_args.routing_override = None