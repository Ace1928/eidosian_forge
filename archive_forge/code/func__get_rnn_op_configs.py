import copy
import operator
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized.reference as nnqr
from collections import namedtuple
from typing import Callable, Dict, List, Union
from .backend_config import (
from ..fuser_method_mappings import (
def _get_rnn_op_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    rnn_op_configs = []
    for rnn_op, ref_rnn_op in [(nn.GRUCell, nnqr.GRUCell), (nn.LSTMCell, nnqr.LSTMCell), (nn.RNNCell, nnqr.RNNCell), (nn.LSTM, nnqr.LSTM), (nn.GRU, nnqr.GRU)]:
        rnn_op_configs.append(BackendPatternConfig(rnn_op).set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT).set_dtype_configs(dtype_configs).set_root_module(rnn_op).set_reference_quantized_module(ref_rnn_op))
    return rnn_op_configs