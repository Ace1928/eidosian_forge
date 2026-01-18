import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Set, Tuple
import numpy as np
import onnx
from onnx import ModelProto, ValueInfoProto, numpy_helper
from ..utils import logging, recurse_getattr
def _get_weights_to_tie(tied_params: List[List[str]], torch_model: 'nn.Module') -> Tuple[List[List[str]]]:
    """
    Separates tied weights from the torch_model in groups for which a tying implementation is (and is not) available.

    Currently, only Embedding and Linear weight sharing the same data can be tied.
    """
    SUPPORTED_DEDUPLICATION_OPS = ('Embedding', 'Linear')
    tied_params_with_op = []
    tied_groups_to_tie = []
    tied_groups_ignored = []
    for params in tied_params:
        tied_params_with_op.append({})
        skip_group = False
        for param_name in params:
            module_name = '.'.join(param_name.split('.')[:-1])
            module = recurse_getattr(torch_model, module_name)
            if module.__class__.__name__ not in SUPPORTED_DEDUPLICATION_OPS:
                skip_group = True
            tied_params_with_op[-1][param_name] = module.__class__.__name__
        if skip_group:
            tied_groups_ignored.append(params)
        else:
            tied_groups_to_tie.append(params)
    return (tied_params_with_op, tied_groups_to_tie, tied_groups_ignored)