from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.dataflow import apis
from googlecloudsdk.api_lib.dataflow import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataflow import dataflow_util
from googlecloudsdk.command_lib.dataflow import job_utils
from googlecloudsdk.core.util import times
def _GetWatermarkSentinelDescription(self, metric):
    """This method gets the description of the watermark sentinel value.

    There are only two watermark sentinel values we care about, -1 represents a
    watermark at kInt64Min. -2 represents a watermark at kInt64Max. This runs
    on the assumption that _IsSentinelWatermark was called first.

    Args:
      metric: A single UpdateMetric returned from the API.
    Returns:
      The sentinel description.
    """
    value = metric.scalar.integer_value
    if value == -1:
        return 'Unknown watermark'
    return 'Max watermark'