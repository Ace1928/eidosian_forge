from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log as sdk_log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
def UpdateLogMetric(metric, description=None, log_filter=None, bucket_name=None, data=None):
    """Updates a LogMetric message given description, filter, and/or data.

  Args:
    metric: LogMetric, the original metric.
    description: str, updated description if any.
    log_filter: str, updated filter for the metric's filter field if any.
    bucket_name: str, the bucket name which ownes the metric.
    data: str, a stream of data read from a config file if any.

  Returns:
    LogMetric, the message representing the updated metric.
  """
    messages = GetMessages()
    if description:
        metric.description = description
    if log_filter:
        metric.filter = log_filter
    if bucket_name:
        metric.bucketName = bucket_name
    if data:
        update_data = yaml.load(data)
        metric_diff = encoding.DictToMessage(update_data, messages.LogMetric)
        for field_name in update_data:
            setattr(metric, field_name, getattr(metric_diff, field_name))
    return metric