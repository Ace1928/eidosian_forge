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
def _create_metric_extractor(metric_name):
    """Returns function that extracts a metric from a session group or a
    session.

    Args:
      metric_name: tensorboard.hparams.MetricName protobuffer. Identifies the
      metric to extract from the session group.
    Returns:
      A function that takes a tensorboard.hparams.SessionGroup or
      tensorborad.hparams.Session protobuffer and returns the value of the metric
      identified by 'metric_name' or None if the value doesn't exist.
    """

    def extractor_fn(session_or_group):
        metric_value = _find_metric_value(session_or_group, metric_name)
        return metric_value.value if metric_value else None
    return extractor_fn