import torch
from torch.fx._symbolic_trace import Tracer
from torch.fx.proxy import Scope
from torch.ao.nn.intrinsic import _FusedModule
from typing import List, Callable
class ScopeContextManager(torch.fx.proxy.ScopeContextManager):

    def __init__(self, scope: Scope, current_module: torch.nn.Module, current_module_path: str):
        super().__init__(scope, Scope(current_module_path, type(current_module)))