import collections
import dataclasses
import operator
import re
from typing import Optional
from google.protobuf import struct_pb2
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import backend_context as backend_context_lib
from tensorboard.plugins.hparams import error
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import metrics
from tensorboard.plugins.hparams import plugin_data_pb2
def _get_hparams_to_include(col_params):
    """Generates the list of hparams to include in the response.

    The determination is based on the `include_in_result` field in ColParam. If
    a ColParam either has `include_in_result: True` or does not specify the
    field at all, then it should be included in the result.

    Args:
      col_params: A collection of `ColParams` protos.

    Returns:
      A list of names of hyperparameters to include in the response.
    """
    hparams_to_include = []
    for col_param in col_params:
        if col_param.HasField('include_in_result') and (not col_param.include_in_result):
            continue
        if col_param.hparam:
            hparams_to_include.append(col_param.hparam)
    return hparams_to_include