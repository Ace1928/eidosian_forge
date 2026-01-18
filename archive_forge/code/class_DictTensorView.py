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
class DictTensorView(TensorView, ABC):
    """
    The DictTensorView abstract class handles the dictionary aspect of the tensor representation,
    which is shared across multiple backends.
    The tensor is contained in the following data structure:
    `Dict[variable_id, Dict[parameter_id, tensor]]`, with the outer dict handling
    the variable offset, and the inner dict handling the parameter offset.
    Subclasses have to implement the implementation of the tensor, as well
    as the tensor operations.
    """

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

    def combine_potentially_none(self, a: dict | None, b: dict | None) -> dict | None:
        """
        Adds the tensor a to b if they are both not none.
        If a (b) is not None but b (a) is None, returns a (b).
        Returns None if both a and b are None.
        """
        if a is None and b is None:
            return None
        elif a is not None and b is None:
            return a
        elif a is None and b is not None:
            return b
        else:
            return self.add_dicts(a, b)

    @staticmethod
    @abstractmethod
    def add_tensors(a: Any, b: Any) -> Any:
        """
        Returns element-wise addition of two tensors of the same type.
        """
        pass

    @staticmethod
    @abstractmethod
    def tensor_type():
        """
        Returns the type of the underlying tensor
        """
        pass

    def add_dicts(self, a: dict, b: dict) -> dict:
        """
        Addition for dict-based tensors.
        """
        res = {}
        keys_a = set(a.keys())
        keys_b = set(b.keys())
        intersect = keys_a & keys_b
        for key in intersect:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                res[key] = self.add_dicts(a[key], b[key])
            elif isinstance(a[key], self.tensor_type()) and isinstance(b[key], self.tensor_type()):
                res[key] = self.add_tensors(a[key], b[key])
            else:
                raise ValueError(f'Values must either be dicts or {self.tensor_type()}.')
        for key in keys_a - intersect:
            res[key] = a[key]
        for key in keys_b - intersect:
            res[key] = b[key]
        return res