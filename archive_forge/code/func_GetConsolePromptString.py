from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six  # pylint: disable=unused-import
from six.moves import filter  # pylint:disable=redefined-builtin
from six.moves import map  # pylint:disable=redefined-builtin
def GetConsolePromptString(queue_string):
    """Parses a full queue reference and returns an abridged version.

  Args:
    queue_string: A full qualifying path for a queue which includes project and
      location, e.g. projects/PROJECT/locations/LOCATION/queues/QUEUE

  Returns:
    A shortened string for the full queue ref which has only the location and
    the queue (LOCATION/QUEUE). For example:
      'projects/myproject/location/us-east1/queue/myqueue' => 'us-east1/myqueue'
  """
    match = re.match('projects\\/.*\\/locations\\/(?P<location>.*)\\/queues\\/(?P<queue>.*)', queue_string)
    if match:
        return '{}/{}'.format(match.group('location'), match.group('queue'))
    return queue_string