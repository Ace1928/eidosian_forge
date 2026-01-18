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
def _create_filter(col_param, extractor):
    """Creates a filter for the given col_param and extractor.

    Args:
      col_param: A tensorboard.hparams.ColParams object identifying the column
        and describing the filter to apply.
      extractor: A function that extract the column value identified by
        'col_param' from a tensorboard.hparams.SessionGroup protobuffer.
    Returns:
      A boolean function taking a tensorboard.hparams.SessionGroup protobuffer
      returning True if the session group passes the filter described by
      'col_param'. If col_param does not specify a filter (i.e. any session
      group passes) returns None.
    """
    include_missing_values = not col_param.exclude_missing_values
    if col_param.HasField('filter_regexp'):
        value_filter_fn = _create_regexp_filter(col_param.filter_regexp)
    elif col_param.HasField('filter_interval'):
        value_filter_fn = _create_interval_filter(col_param.filter_interval)
    elif col_param.HasField('filter_discrete'):
        value_filter_fn = _create_discrete_set_filter(col_param.filter_discrete)
    elif include_missing_values:
        return None
    else:
        value_filter_fn = lambda _: True

    def filter_fn(session_group):
        value = extractor(session_group)
        if value is None:
            return include_missing_values
        return value_filter_fn(value)
    return filter_fn