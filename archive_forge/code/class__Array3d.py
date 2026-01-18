import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
class _Array3d(_Array):
    """3-dimensional array of floats"""

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield (lambda v: validate_array(v, ndim=3))

    @property
    @abstractmethod
    def ndim(self) -> Literal[3]:
        ...

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int, int]:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Array2d]:
        ...

    @abstractmethod
    def astype(self, dtype: DTypes, order: str=..., casting: str=..., subok: bool=..., copy: bool=...) -> 'Array3d':
        ...

    @abstractmethod
    def __add__(self: SelfT, other: Union[float, int, Array1d, Array2d, 'Array3d']) -> SelfT:
        ...

    @abstractmethod
    def __sub__(self: SelfT, other: Union[float, int, Array1d, Array2d, 'Array3d']) -> SelfT:
        ...

    @abstractmethod
    def __mul__(self: SelfT, other: Union[float, int, Array1d, Array2d, 'Array3d']) -> SelfT:
        ...

    @abstractmethod
    def __pow__(self: SelfT, other: Union[float, int, Array1d, Array2d, 'Array3d']) -> SelfT:
        ...

    @abstractmethod
    def __matmul__(self: SelfT, other: Union[float, int, Array1d, Array2d, 'Array3d']) -> SelfT:
        ...

    @abstractmethod
    def __iadd__(self, other: Union[float, int, Array1d, Array2d, 'Array3d']):
        ...

    @abstractmethod
    def __isub__(self, other: Union[float, int, Array1d, Array2d, 'Array3d']):
        ...

    @abstractmethod
    def __imul__(self, other: Union[float, int, Array1d, Array2d, 'Array3d']):
        ...

    @abstractmethod
    def __ipow__(self, other: Union[float, int, Array1d, Array2d, 'Array3d']):
        ...

    @overload
    @abstractmethod
    def argmax(self, keepdims: Fal=False, axis: int=-1, out: Optional[_Array]=None) -> Ints2d:
        ...

    @overload
    @abstractmethod
    def argmax(self, keepdims: Tru, axis: int=-1, out: Optional[_Array]=None) -> 'Ints3d':
        ...

    @abstractmethod
    def argmax(self, keepdims: bool=False, axis: int=-1, out: Optional[_Array]=None) -> Union[Ints2d, 'Ints3d']:
        ...