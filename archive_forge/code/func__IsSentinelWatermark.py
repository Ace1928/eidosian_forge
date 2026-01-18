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
def _IsSentinelWatermark(self, metric):
    """This returns true if the metric is a watermark with a sentinel value.

    Args:
      metric: A single UpdateMetric returned from the API.
    Returns:
      True if the metric is a sentinel value, false otherwise.
    """
    if not dataflow_util.DATAFLOW_METRICS_RE.match(metric.name.origin):
        return False
    if not dataflow_util.WINDMILL_WATERMARK_RE.match(metric.name.name):
        return False
    return metric.scalar.integer_value == -1 or metric.scalar.integer_value == -2