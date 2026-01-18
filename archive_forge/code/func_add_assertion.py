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
def add_assertion(self, e: Union[z3.BoolRef, sympy.Basic]) -> None:
    if isinstance(e, sympy.Basic):
        self._check_freesymbols(e)
        ref = self.to_z3_boolean_expr(e)
    else:
        ref = e
    assert isinstance(ref, z3.BoolRef)
    if ref not in self._assertions:
        log.debug('add assertion: %s', z3str(ref))
    self._assertions.add(ref)