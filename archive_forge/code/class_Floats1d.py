import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
class Floats1d(_Array1d, _Floats):
    """1-dimensional array of floats."""
    T: 'Floats1d'

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield (lambda v: validate_array(v, ndim=1, dtype='f'))

    @abstractmethod
    def __iter__(self) -> Iterator[float]:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _1_KeyScalar) -> float:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _1_Key1d) -> 'Floats1d':
        ...

    @abstractmethod
    def __getitem__(self, key: _1_AllKeys) -> _F1_AllReturns:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _1_KeyScalar, value: float) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _1_Key1d, value: 'Floats1d') -> None:
        ...

    @abstractmethod
    def __setitem__(self, key: _1_AllKeys, _F1_AllReturns) -> None:
        ...

    @overload
    @abstractmethod
    def cumsum(self, *, keepdims: Tru, axis: Optional[OneAx]=None, out: Optional['Floats1d']=None) -> 'Floats1d':
        ...

    @overload
    @abstractmethod
    def cumsum(self, *, keepdims: Fal, axis: Optional[OneAx]=None, out: Optional['Floats1d']=None) -> 'Floats1d':
        ...

    @abstractmethod
    def cumsum(self, *, keepdims: bool=False, axis: _1_AllAx=None, out: Optional['Floats1d']=None) -> 'Floats1d':
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Tru, axis: Optional[OneAx]=None, out: Optional['Floats1d']=None) -> 'Floats1d':
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal, axis: Optional[OneAx]=None, out=None) -> float:
        ...

    @abstractmethod
    def sum(self, *, keepdims: bool=False, axis: _1_AllAx=None, out: Optional['Floats1d']=None) -> _1F_ReduceResults:
        ...