import torch.fx as fx
from torch.fx.node import Argument, Target
from torch.nn.utils.fusion import fuse_conv_bn_eval
from typing import Type, Dict, Any, Tuple, Iterable, Optional, List, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp
import copy
from collections import defaultdict
import torch.utils.mkldnn as th_mkldnn
import operator
import time
import logging
from enum import Enum
class DropoutRemover(torch.fx.Transformer):

    def call_module(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        if isinstance(self.submodules[target], nn.Dropout):
            assert len(args) == 1
            return args[0]
        else:
            return super().call_module(target, args, kwargs)