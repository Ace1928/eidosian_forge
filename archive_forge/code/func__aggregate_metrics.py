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
def _aggregate_metrics(self, session_group):
    """Sets the metrics of the group based on aggregation_type."""
    if self._request.aggregation_type == api_pb2.AGGREGATION_AVG or self._request.aggregation_type == api_pb2.AGGREGATION_UNSET:
        _set_avg_session_metrics(session_group)
    elif self._request.aggregation_type == api_pb2.AGGREGATION_MEDIAN:
        _set_median_session_metrics(session_group, self._request.aggregation_metric)
    elif self._request.aggregation_type == api_pb2.AGGREGATION_MIN:
        _set_extremum_session_metrics(session_group, self._request.aggregation_metric, min)
    elif self._request.aggregation_type == api_pb2.AGGREGATION_MAX:
        _set_extremum_session_metrics(session_group, self._request.aggregation_metric, max)
    else:
        raise error.HParamsError('Unknown aggregation_type in request: %s' % self._request.aggregation_type)