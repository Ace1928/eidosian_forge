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
def _build_data_provider_sort_item(col_param):
    """Builds HyperparameterSort from ColParam.

    Args:
      col_param: ColParam that possibly contains sort information.

    Returns:
      None if col_param does not specify sort information.
    """
    if col_param.order == api_pb2.ORDER_UNSPECIFIED:
        return None
    sort_direction = provider.HyperparameterSortDirection.ASCENDING if col_param.order == api_pb2.ORDER_ASC else provider.HyperparameterSortDirection.DESCENDING
    return provider.HyperparameterSort(hyperparameter_name=col_param.hparam, sort_direction=sort_direction)