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
def _Format(self, metric):
    """Performs extra formatting for sentinel values or otherwise.

    Args:
      metric: A single UpdateMetric returned from the API.
    Returns:
      The formatted metric.
    """
    if self._IsSentinelWatermark(metric):
        metric.scalar.string_value = self._GetWatermarkSentinelDescription(metric)
        metric.scalar.reset('integer_value')
    return metric