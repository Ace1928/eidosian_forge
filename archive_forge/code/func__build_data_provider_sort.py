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
def _build_data_provider_sort(col_params):
    """Builds HyperparameterSorts from ColParams."""
    sort = []
    for col_param in col_params:
        sort_item = _build_data_provider_sort_item(col_param)
        if sort_item is None:
            continue
        sort.append(sort_item)
    return sort