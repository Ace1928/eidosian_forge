import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
class _Array1d(_Array):
    """1-dimensional array."""

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield (lambda v: validate_array(v, ndim=1))

    @property
    @abstractmethod
    def ndim(self) -> Literal[1]:
        ...

    @property
    @abstractmethod
    def shape(self) -> Tuple[int]:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Union[float, int]]:
        ...

    @abstractmethod
    def astype(self, dtype: DTypes, order: str=..., casting: str=..., subok: bool=..., copy: bool=...) -> '_Array1d':
        ...

    @abstractmethod
    def flatten(self: SelfT, order: str=...) -> SelfT:
        ...

    @abstractmethod
    def ravel(self: SelfT, order: str=...) -> SelfT:
        ...

    @abstractmethod
    def __add__(self: SelfT, other: Union[float, int, 'Array1d']) -> SelfT:
        ...

    @abstractmethod
    def __sub__(self: SelfT, other: Union[float, int, 'Array1d']) -> SelfT:
        ...

    @abstractmethod
    def __mul__(self: SelfT, other: Union[float, int, 'Array1d']) -> SelfT:
        ...

    @abstractmethod
    def __pow__(self: SelfT, other: Union[float, int, 'Array1d']) -> SelfT:
        ...

    @abstractmethod
    def __matmul__(self: SelfT, other: Union[float, int, 'Array1d']) -> SelfT:
        ...

    @abstractmethod
    def __iadd__(self, other: Union[float, int, 'Array1d']):
        ...

    @abstractmethod
    def __isub__(self, other: Union[float, int, 'Array1d']):
        ...

    @abstractmethod
    def __imul__(self, other: Union[float, int, 'Array1d']):
        ...

    @abstractmethod
    def __ipow__(self, other: Union[float, int, 'Array1d']):
        ...

    @overload
    @abstractmethod
    def argmax(self, keepdims: Fal=False, axis: int=-1, out: Optional[_Array]=None) -> int:
        ...

    @overload
    @abstractmethod
    def argmax(self, keepdims: Tru, axis: int=-1, out: Optional[_Array]=None) -> 'Ints1d':
        ...

    @abstractmethod
    def argmax(self, keepdims: bool=False, axis: int=-1, out: Optional[_Array]=None) -> Union[int, 'Ints1d']:
        ...

    @overload
    @abstractmethod
    def mean(self, keepdims: Tru, axis: int=-1, dtype: Optional[DTypes]=None, out: Optional['Floats1d']=None) -> 'Floats1d':
        ...

    @overload
    @abstractmethod
    def mean(self, keepdims: Fal=False, axis: int=-1, dtype: Optional[DTypes]=None, out: Optional['Floats1d']=None) -> float:
        ...

    @abstractmethod
    def mean(self, keepdims: bool=False, axis: int=-1, dtype: Optional[DTypes]=None, out: Optional['Floats1d']=None) -> Union['Floats1d', float]:
        ...