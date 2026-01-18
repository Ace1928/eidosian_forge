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
def _set_extremum_session_metrics(session_group, aggregation_metric, extremum_fn):
    """Sets the metrics for session_group to those of its "extremum session".

    The extremum session is the session in session_group with the extremum value
    of the metric given by 'aggregation_metric'. The extremum is taken over the
    subset of sessions in the group whose 'aggregation_metric' was measured
    at the largest training step among the sessions in the group.

    Args:
      session_group: A SessionGroup protobuffer.
      aggregation_metric: A MetricName protobuffer.
      extremum_fn: callable. Must be either 'min' or 'max'. Determines the type of
        extremum to compute.
    """
    measurements = _measurements(session_group, aggregation_metric)
    ext_session = extremum_fn(measurements, key=operator.attrgetter('metric_value.value')).session_index
    del session_group.metric_values[:]
    session_group.metric_values.MergeFrom(session_group.sessions[ext_session].metric_values)