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
def _create_regexp_filter(regex):
    """Returns a boolean function that filters strings based on a regular exp.

    Args:
      regex: A string describing the regexp to use.
    Returns:
      A function taking a string and returns True if any of its substrings
      matches regex.
    """
    compiled_regex = re.compile(regex)

    def filter_fn(value):
        if not isinstance(value, str):
            raise error.HParamsError('Cannot use a regexp filter for a value of type %s. Value: %s' % (type(value), value))
        return re.search(compiled_regex, value) is not None
    return filter_fn