import functools
import logging
import math
import operator
import sympy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch._dynamo.exc import TorchDynamoException
from torch.fx.node import Argument, Target
from torch.utils._sympy.interp import sympy_interp
from torch.fx.experimental import _config as config
def _check_freesymbols(self, e: sympy.Basic) -> None:
    for s in e.free_symbols:
        assert isinstance(s, sympy.Symbol)
        self.z3var(s)