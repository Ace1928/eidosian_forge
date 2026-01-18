import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
class _Array2d(_Array):

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield (lambda v: validate_array(v, ndim=2))

    @property
    @abstractmethod
    def ndim(self) -> Literal[2]:
        ...

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Array1d]:
        ...

    @abstractmethod
    def astype(self, dtype: DTypes, order: str=..., casting: str=..., subok: bool=..., copy: bool=...) -> 'Array2d':
        ...

    @abstractmethod
    def __add__(self: ArrayT, other: Union[float, int, Array1d, 'Array2d']) -> ArrayT:
        ...

    @abstractmethod
    def __sub__(self: ArrayT, other: Union[float, int, Array1d, 'Array2d']) -> ArrayT:
        ...

    @abstractmethod
    def __mul__(self: ArrayT, other: Union[float, int, Array1d, 'Array2d']) -> ArrayT:
        ...

    @abstractmethod
    def __pow__(self: ArrayT, other: Union[float, int, Array1d, 'Array2d']) -> ArrayT:
        ...

    @abstractmethod
    def __matmul__(self: ArrayT, other: Union[float, int, Array1d, 'Array2d']) -> ArrayT:
        ...

    @abstractmethod
    def __iadd__(self, other: Union[float, int, Array1d, 'Array2d']):
        ...

    @abstractmethod
    def __isub__(self, other: Union[float, int, Array1d, 'Array2d']):
        ...

    @abstractmethod
    def __imul__(self, other: Union[float, int, Array1d, 'Array2d']):
        ...

    @abstractmethod
    def __ipow__(self, other: Union[float, int, Array1d, 'Array2d']):
        ...

    @overload
    @abstractmethod
    def argmax(self, keepdims: Fal=False, axis: int=-1, out: Optional[_Array]=None) -> Ints1d:
        ...

    @overload
    @abstractmethod
    def argmax(self, keepdims: Tru, axis: int=-1, out: Optional[_Array]=None) -> 'Ints2d':
        ...

    @abstractmethod
    def argmax(self, keepdims: bool=False, axis: int=-1, out: Optional[_Array]=None) -> Union[Ints1d, 'Ints2d']:
        ...

    @overload
    @abstractmethod
    def mean(self, keepdims: Fal=False, axis: int=-1, dtype: Optional[DTypes]=None, out: Optional['Floats2d']=None) -> Floats1d:
        ...

    @overload
    @abstractmethod
    def mean(self, keepdims: Tru, axis: int=-1, dtype: Optional[DTypes]=None, out: Optional['Floats2d']=None) -> 'Floats2d':
        ...

    @abstractmethod
    def mean(self, keepdims: bool=False, axis: int=-1, dtype: Optional[DTypes]=None, out: Optional['Floats2d']=None) -> Union['Floats2d', Floats1d]:
        ...