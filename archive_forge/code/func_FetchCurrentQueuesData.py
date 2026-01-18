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
def FetchCurrentQueuesData(tasks_api):
    """Fetches the current queues data stored in the database.

  Args:
    tasks_api: api_lib.tasks.<Alpha|Beta|GA>ApiAdapter, Cloud Tasks API needed
      for doing queue based operations.

  Returns:
    A dictionary with queue names as keys and corresponding protobuf Queue
    objects as values apis.cloudtasks.<ver>.cloudtasks_<ver>_messages.Queue
  """
    queues_client = tasks_api.queues
    app_location = app.ResolveAppLocation(parsers.ParseProject())
    region_ref = parsers.ParseLocation(app_location)
    all_queues_in_db_dict = {os.path.basename(x.name): x for x in queues_client.List(region_ref)}
    return all_queues_in_db_dict