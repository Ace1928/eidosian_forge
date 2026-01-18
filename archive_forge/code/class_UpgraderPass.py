import logging
from collections import defaultdict
from typing import Tuple, Dict, Optional, List
import torch
from torch._export import export
from torch._export.pass_base import _ExportPassBase
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue
from torch._subclasses import FakeTensor
from torch.fx.node import Target, Argument
from torch.library import Library
from torch.utils._pytree import tree_unflatten
import torch._export.exported_program as ep
import re
class UpgraderPass(_ExportPassBase):

    def __init__(self, old_target: Target, new_target: Target):
        super().__init__()
        self.old_target = old_target
        self.new_target = new_target

    def call_operator(self, op, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], meta: NodeMetadata) -> ProxyValue:
        if op == self.old_target:
            return super().call_operator(self.new_target, args, kwargs, meta)
        return super().call_operator(op, args, kwargs, meta)