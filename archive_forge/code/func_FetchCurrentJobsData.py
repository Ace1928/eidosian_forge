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
def FetchCurrentJobsData(scheduler_api):
    """Fetches the current jobs data stored in the database.

  Args:
    scheduler_api: api_lib.scheduler.<Alpha|Beta|GA>ApiAdapter, Cloud Scheduler
      API needed for doing jobs based operations.

  Returns:
    A list of currently existing jobs in the backend.
  """
    jobs_client = scheduler_api.jobs
    app_location = app.ResolveAppLocation(parsers.ParseProject(), locations_client=scheduler_api.locations)
    region_ref = parsers.ParseLocation(app_location).RelativeName()
    return list((x for x in jobs_client.List(region_ref)))