import copy
import operator
from copy import deepcopy
from typing import cast, Dict, List, Optional, Union
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from torch.export.exported_program import (
from torch.fx import GraphModule
from .utils import _check_input_constraints_pre_hook
def _add_spec(gm: torch.fx.GraphModule, spec) -> str:
    i = 0
    while hasattr(gm, f'_spec_{i}'):
        i += 1
    name = f'_spec_{i}'
    setattr(gm, name, spec)
    return name