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
def _create_discrete_set_filter(discrete_set):
    """Returns a function that checks whether a value belongs to a set.

    Args:
      discrete_set: A list of objects representing the set.
    Returns:
      A function taking an object and returns True if its in the set. Membership
      is tested using the Python 'in' operator (thus, equality of distinct
      objects is computed using the '==' operator).
    """

    def filter_fn(value):
        return value in discrete_set
    return filter_fn