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
def _create_interval_filter(interval):
    """Returns a function that checkes whether a number belongs to an interval.

    Args:
      interval: A tensorboard.hparams.Interval protobuf describing the interval.
    Returns:
      A function taking a number (float or int) that returns True if the number
      belongs to (the closed) 'interval'.
    """

    def filter_fn(value):
        if not isinstance(value, (int, float)):
            raise error.HParamsError('Cannot use an interval filter for a value of type: %s, Value: %s' % (type(value), value))
        return interval.min_value <= value and value <= interval.max_value
    return filter_fn