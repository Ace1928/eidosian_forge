import ast
import builtins
import dis
import enum
import inspect
import re
import typing
import warnings
from textwrap import dedent
from typing import Type
import torch
from torch._C import (
from torch._sources import get_source_lines_and_file
from .._jit_internal import (  # type: ignore[attr-defined]
from ._state import _get_script_class
from torch._ops import OpOverloadPacket
class EvalEnv:
    env = {'torch': Module('torch', {'Tensor': torch.Tensor}), 'Tensor': torch.Tensor, 'typing': Module('typing', {'Tuple': Tuple}), 'Tuple': Tuple, 'List': List, 'Dict': Dict, 'Optional': Optional, 'Union': Union, 'Future': Future, 'Await': _Await}

    def __init__(self, rcb):
        self.rcb = rcb
        if torch.distributed.rpc.is_available():
            self.env['RRef'] = RRef

    def __getitem__(self, name):
        if name in self.env:
            return self.env[name]
        if self.rcb is not None:
            return self.rcb(name)
        return getattr(builtins, name, None)