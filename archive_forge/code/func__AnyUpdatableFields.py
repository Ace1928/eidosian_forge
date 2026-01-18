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
def _AnyUpdatableFields(args):
    """Check whether the queue has any changed attributes based on args provided.

  Args:
    args: argparse.Namespace, A placeholder args namespace built to pass on
      forwards to Cloud Tasks API.

  Returns:
    True if any of the queue attributes have changed from the attributes stored
    in the backend, False otherwise.
  """
    modifiable_args = [x for x in args._specified_args if x not in ('name', 'type')]
    return True if modifiable_args else False