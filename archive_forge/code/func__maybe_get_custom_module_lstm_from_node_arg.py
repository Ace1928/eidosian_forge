import copy
import torch
import torch.nn as nn
from torch.ao.quantization import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.fake_quantize import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.qconfig import (
from torch.ao.quantization.stubs import DeQuantStub
from torch.ao.quantization.utils import (
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.fx import GraphModule, map_arg
from torch.fx.graph import (
from .custom_config import PrepareCustomConfig
from ._decomposed import quantized_decomposed_lib  # noqa: F401
from typing import Callable, Optional, List, Dict, Any, Set, Tuple, Union, Type
from dataclasses import dataclass
from collections import namedtuple
import operator
import warnings
def _maybe_get_custom_module_lstm_from_node_arg(arg: Node, named_modules: Dict[str, torch.nn.Module]) -> Optional[Node]:
    """
    Given an argument of a node, if the argument refers to the path through which the node
    is a consumer of custom module LSTM, return the custom module LSTM node, or None otherwise.

    This is used to determine whether a node is a consumer of custom module LSTM, and, if so,
    skip inserting input observers for this node. This is because custom module LSTM produces
    quantized outputs, so inserting an input observer for the consumer of custom module LSTM
    would unnecessarily quantize the outputs again.

      lstm -> consumer

    In practice, however, custom module LSTM outputs a tuple (output, (hidden0, hidden1)) with
    DeQuantStubs attached to each internal node (see `_insert_dequant_stubs_for_custom_module_lstm_output`).
    This tuple can be consumed in one of four ways:

      lstm -> getitem -> DeQuantStub -> consumer                       # consume lstm[0]
      lstm -> getitem -> getitem -> DeQuantStub -> tuple -> consumer   # consume lstm[1]
      lstm -> getitem -> getitem -> DeQuantStub -> consumer            # consume lstm[1][0] or lstm[1][1]
      lstm -> getitem -> DeQuantStub -> tuple -> consumer              # consume lstm

    Thus, we must match against the above patterns instead of simply checking the parent node
    to determine whether this node is a consumer of a custom module LSTM.
    """

    def match_dq(a):
        return isinstance(_get_module(a, named_modules), DeQuantStub)

    def match_lstm(a):
        return _is_custom_module_lstm(a, named_modules)

    def match_getitem(a):
        return a.op == 'call_function' and a.target == operator.getitem

    def match_tuple(a):
        return a.op == 'call_function' and a.target == tuple

    def _match_pattern(match_pattern: List[Callable]) -> Optional[Node]:
        """
        Traverse up the graph and match the args one by one.
        If there is a match, return the last matched node, or None otherwise.
        """
        a = arg
        for i, match in enumerate(match_pattern):
            if not match(a):
                return None
            if i < len(match_pattern) - 1:
                if match == match_tuple:
                    a = a.args[0][0]
                else:
                    a = a.args[0]
        return a
    all_match_patterns = [[match_dq, match_getitem, match_lstm], [match_tuple, match_dq, match_getitem, match_getitem, match_lstm], [match_dq, match_getitem, match_getitem, match_lstm], [match_tuple, match_dq, match_getitem, match_lstm]]
    for p in all_match_patterns:
        matched_node = _match_pattern(p)
        if matched_node is not None:
            return matched_node
    return None