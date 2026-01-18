import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
def _sfdp_scale_factor_check(scale_factor_op):

    def fn(match):
        scale_factor_node = filter_nodes(match.nodes, scale_factor_op)[0]
        scale_factor = scale_factor_node.args[1]
        if not isinstance(scale_factor, (float, int)):
            return False
        return _sfdp_params_check(match)
    return fn