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
class TensorView(ABC):
    """
    A TensorView represents the tensors for A and b, which are of shape
    rows x var_length x param_size_plus_one and rows x 1 x param_size_plus_one, respectively.
    The class facilitates the application of the CanonBackend functions.
    """

    def __init__(self, variable_ids: set[int] | None, tensor: Any, is_parameter_free: bool, param_size_plus_one: int, id_to_col: dict[int, int], param_to_size: dict[int, int], param_to_col: dict[int, int], var_length: int):
        self.variable_ids = variable_ids if variable_ids is not None else None
        self.tensor = tensor
        self.is_parameter_free = is_parameter_free
        self.param_size_plus_one = param_size_plus_one
        self.id_to_col = id_to_col
        self.param_to_size = param_to_size
        self.param_to_col = param_to_col
        self.var_length = var_length

    def __iadd__(self, other: TensorView) -> TensorView:
        assert isinstance(other, self.__class__)
        self.variable_ids = self.variable_ids | other.variable_ids
        self.tensor = self.combine_potentially_none(self.tensor, other.tensor)
        self.is_parameter_free = self.is_parameter_free and other.is_parameter_free
        return self

    @staticmethod
    @abstractmethod
    def combine_potentially_none(a: Any | None, b: Any | None) -> Any | None:
        """
        Adds the tensor a to b if they are both not none.
        If a (b) is not None but b (a) is None, returns a (b).
        Returns None if both a and b are None.
        """
        pass

    @classmethod
    def get_empty_view(cls, param_size_plus_one: int, id_to_col: dict[int, int], param_to_size: dict[int, int], param_to_col: dict[int, int], var_length: int) -> TensorView:
        """
        Return a TensorView that has shape information, but no data.
        """
        return cls(None, None, True, param_size_plus_one, id_to_col, param_to_size, param_to_col, var_length)

    @staticmethod
    def is_constant_data(variable_ids: set[int]) -> bool:
        """
        Does the TensorView only contain constant data?
        """
        return variable_ids == {Constant.ID.value}

    @property
    @abstractmethod
    def rows(self) -> int:
        """
        Number of rows of the TensorView.
        """
        pass

    @abstractmethod
    def get_tensor_representation(self, row_offset: int, total_rows: int) -> TensorRepresentation:
        """
        Returns [A b].
        """
        pass

    @abstractmethod
    def select_rows(self, rows: np.ndarray) -> None:
        """
        Select 'rows' from tensor.
        """
        pass

    @abstractmethod
    def apply_all(self, func: Callable) -> None:
        """
        Apply 'func' across all variables and parameter slices.
        """
        pass

    @abstractmethod
    def create_new_tensor_view(self, variable_ids: set[int], tensor: Any, is_parameter_free: bool) -> TensorView:
        """
        Create new TensorView with same shape information as self, but new data.
        """
        pass