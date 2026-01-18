import inspect
import operator
from typing import Dict, List, Optional, Tuple, cast
from torch.distributed.nn import RemoteModule
import torch.fx
import torch.nn as nn
from . import PipelineModulesGraph
class RemoteModuleTracer(torch.fx.Tracer):

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, RemoteModule):
            return True
        return False