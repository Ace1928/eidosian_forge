import copy
import logging
from typing import List, Optional
import torch
import torch.nn as nn
from torch._dynamo.utils import detect_fake_mode
from torch._utils_internal import print_graph
from torch.fx.experimental.optimization import (
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_conv_bn_weights
from .. import config
from ..fx_utils import matches_module_function_pattern
from ..pattern_matcher import (
from ..utils import is_cpu_device
from .group_batch_fusion import group_batch_fusion_passes
from .misc_patterns import numpy_compat_normalization
class NormalizedLinearNode:

    def __init__(self, node: torch.fx.Node) -> None:
        assert node.op == 'call_function'
        assert node.target in [torch.nn.functional.linear]
        self.node: torch.fx.Node = node

    def get_input(self) -> torch.fx.Node:
        if len(self.node.args) > 0:
            return self.node.args[0]
        else:
            return self.node.kwargs['input']

    def get_weight(self) -> torch.fx.Node:
        if len(self.node.args) > 1:
            return self.node.args[1]
        else:
            return self.node.kwargs['weight']

    def get_bias(self) -> torch.fx.Node:
        if len(self.node.args) > 2:
            return self.node.args[2]
        else:
            return self.node.kwargs['bias'] if 'bias' in self.node.kwargs else None