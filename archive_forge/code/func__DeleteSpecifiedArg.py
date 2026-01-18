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
def _DeleteSpecifiedArg(cloud_task_args, key):
    """Deletes the specified key in the namespace provided.

  Args:
    cloud_task_args: argparse.Namespace, A placeholder args namespace built to
      pass on forwards to Cloud Tasks API.
    key: The attribute key we are trying to set.
  """
    del cloud_task_args._specified_args[key]