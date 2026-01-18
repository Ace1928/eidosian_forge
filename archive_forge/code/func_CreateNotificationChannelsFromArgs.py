from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def CreateNotificationChannelsFromArgs(args, messages):
    """Builds a notification channel message from args.

  Args:
    args: Flags provided by the user.
    messages: Object containing information about all message types allowed.

  Returns:
     The notification channels corresponding to the Prometheus alert manager
     YAML file provided. In the case that no file is specified, the default
     behavior is to return an empty list.
  """
    if args.IsSpecified('channels_from_prometheus_alertmanager_yaml'):
        alert_manager_yaml = args.channels_from_prometheus_alertmanager_yaml
        channels = NotificationChannelMessageFromString(alert_manager_yaml, messages)
    else:
        channels = []
    return channels