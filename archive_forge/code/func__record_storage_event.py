from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task_util
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def _record_storage_event(metric, value=0):
    """Common code for processing an event.

  Args:
    metric (str): The metric being recorded.
    value (mixed): The value being recorded.
  """
    command_name = properties.VALUES.metrics.command_name.Get()
    metrics.CustomKeyValue(command_name, 'Storage-' + metric, value)