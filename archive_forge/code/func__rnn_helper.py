import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
def _rnn_helper(input, hidden, params, has_biases, num_layers, dropout, train, bidirectional, batch_first, layer_fn):
    input = input.transpose(0, 1) if batch_first else input
    final_hiddens = []
    for i in range(num_layers):
        cur_params, cur_hidden, bidir_params, bidir_hidden = params_hiddens(params, hidden, i, bidirectional)
        dropout = dropout if train and num_layers < i - 1 else 0.0
        fwd_inp, fwd_hidden = layer_fn(input, cur_hidden, cur_params, has_biases)
        final_hiddens.append(fwd_hidden)
        if bidirectional:
            bwd_inp, bwd_hidden = layer_fn(input, bidir_hidden, bidir_params, has_biases, reverse=True)
            final_hiddens.append(bwd_hidden)
        if bidirectional:
            input = torch.cat([fwd_inp, bwd_inp], fwd_inp.dim() - 1)
        else:
            input = fwd_inp
        if dropout != 0 and train and (i < num_layers - 1):
            input = torch.dropout(input, dropout, train=True)
    input = input.transpose(0, 1) if batch_first else input
    return (input, final_hiddens)