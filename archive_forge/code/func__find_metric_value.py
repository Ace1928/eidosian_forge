import collections
import dataclasses
import operator
import re
from typing import Optional
from google.protobuf import struct_pb2
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import error
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import metrics
def _find_metric_value(session_or_group, metric_name):
    """Returns the metric_value for a given metric in a session or session
    group.

    Args:
      session_or_group: A Session protobuffer or SessionGroup protobuffer.
      metric_name: A MetricName protobuffer. The metric to search for.
    Returns:
      A MetricValue protobuffer representing the value of the given metric or
      None if no such metric was found in session_or_group.
    """
    for metric_value in session_or_group.metric_values:
        if metric_value.name.tag == metric_name.tag and metric_value.name.group == metric_name.group:
            return metric_value