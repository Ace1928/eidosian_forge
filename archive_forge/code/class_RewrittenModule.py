import ast
import inspect
import textwrap
import copy
import functools
from types import FunctionType
from typing import cast, Union, Callable, Dict, Optional, Any
from torch.fx._symbolic_trace import Tracer
from torch.fx.graph import Graph
from torch._sources import normalize_source_lines
import torch
class RewrittenModule(torch.nn.Module):

    def __init__(self, orig):
        super().__init__()
        for k, v in orig.__dict__.items():
            if isinstance(v, torch.nn.Module):
                self.__dict__[k] = copy.copy(rewrite_module(v))
            else:
                self.__dict__[k] = copy.copy(v)