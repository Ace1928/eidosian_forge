from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable
import numpy as np
import scipy.sparse as sp
from scipy.signal import convolve
from cvxpy.lin_ops import LinOp
from cvxpy.settings import (
def accumulate_over_variables(self, func: Callable, is_param_free_function: bool) -> TensorView:
    """
        Apply 'func' to A and b.
        If 'func' is a parameter free function, then we can apply it to all parameter slices
        (including the slice that contains non-parameter constants).
        If 'func' is not a parameter free function, we only need to consider the parameter slice
        that contains the non-parameter constants, due to DPP rules.
        """
    for variable_id, tensor in self.tensor.items():
        self.tensor[variable_id] = self.apply_to_parameters(func, tensor) if is_param_free_function else func(tensor[Constant.ID.value])
    self.is_parameter_free = self.is_parameter_free and is_param_free_function
    return self