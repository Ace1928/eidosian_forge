from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from googlecloudsdk.command_lib.logs import stream
import six
def _LogFilters(name, task_name):
    """Returns filters for log fetcher to use.

  Args:
    name: string id of the entity.
    task_name: String name of task.

  Returns:
    A list of filters to be passed to the logging API.
  """
    filters = ['resource.type="ml_job"', 'resource.labels.job_id="{0}"'.format(name)]
    if task_name:
        filters.append('resource.labels.task_name="{0}"'.format(task_name))
    return filters