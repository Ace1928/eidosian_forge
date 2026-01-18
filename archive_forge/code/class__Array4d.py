import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
class _Array4d(_Array):
    """4-dimensional array."""

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield (lambda v: validate_array(v, ndim=4))

    @property
    @abstractmethod
    def ndim(self) -> Literal[4]:
        ...

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int, int, int]:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Array3d]:
        ...

    @abstractmethod
    def astype(self, dtype: DTypes, order: str=..., casting: str=..., subok: bool=..., copy: bool=...) -> '_Array4d':
        ...

    @abstractmethod
    def __add__(self: SelfT, other: Union[float, int, Array1d, Array2d, Array3d, 'Array4d']) -> SelfT:
        ...

    @abstractmethod
    def __sub__(self: SelfT, other: Union[float, int, Array1d, Array2d, Array3d, 'Array4d']) -> SelfT:
        ...

    @abstractmethod
    def __mul__(self: SelfT, other: Union[float, int, Array1d, Array2d, Array3d, 'Array4d']) -> SelfT:
        ...

    @abstractmethod
    def __pow__(self: SelfT, other: Union[float, int, Array1d, Array2d, Array3d, 'Array4d']) -> SelfT:
        ...

    @abstractmethod
    def __matmul__(self: SelfT, other: Union[float, int, Array1d, Array2d, Array3d, 'Array4d']) -> SelfT:
        ...

    @abstractmethod
    @abstractmethod
    def __iadd__(self, other: Union[float, int, Array1d, Array2d, Array3d, 'Array4d']):
        ...

    @abstractmethod
    def __isub__(self, other: Union[float, int, Array1d, Array2d, Array3d, 'Array4d']):
        ...

    @abstractmethod
    def __imul__(self, other: Union[float, int, Array1d, Array2d, Array3d, 'Array4d']):
        ...

    @abstractmethod
    def __ipow__(self, other: Union[float, int, Array1d, Array2d, Array3d, 'Array4d']):
        ...