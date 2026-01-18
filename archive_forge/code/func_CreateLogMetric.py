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
def CreateLogMetric(metric_name, description=None, log_filter=None, bucket_name=None, data=None):
    """Returns a LogMetric message based on a data stream or a description/filter.

  Args:
    metric_name: str, the name of the metric.
    description: str, a description.
    log_filter: str, the filter for the metric's filter field.
    bucket_name: str, the bucket name which ownes the metric.
    data: str, a stream of data read from a config file.

  Returns:
    LogMetric, the message representing the new metric.
  """
    messages = GetMessages()
    if data:
        contents = yaml.load(data)
        metric_msg = encoding.DictToMessage(contents, messages.LogMetric)
        metric_msg.name = metric_name
    else:
        metric_msg = messages.LogMetric(name=metric_name, description=description, filter=log_filter, bucketName=bucket_name)
    return metric_msg