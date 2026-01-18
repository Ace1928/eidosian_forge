import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
class Ints1d(_Array1d, _Ints):
    """1-dimensional array of ints."""
    T: 'Ints1d'

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield (lambda v: validate_array(v, ndim=1, dtype='i'))

    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _1_KeyScalar) -> int:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _1_Key1d) -> 'Ints1d':
        ...

    @abstractmethod
    def __getitem__(self, key: _1_AllKeys) -> _I1_AllReturns:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _1_KeyScalar, value: int) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _1_Key1d, value: Union[int, 'Ints1d']) -> None:
        ...

    @abstractmethod
    def __setitem__(self, key: _1_AllKeys, _I1_AllReturns) -> None:
        ...

    @overload
    @abstractmethod
    def cumsum(self, *, keepdims: Tru, axis: Optional[OneAx]=None, out: Optional['Ints1d']=None) -> 'Ints1d':
        ...

    @overload
    @abstractmethod
    def cumsum(self, *, keepdims: Fal=False, axis: Optional[OneAx]=None, out: Optional['Ints1d']=None) -> 'Ints1d':
        ...

    @abstractmethod
    def cumsum(self, *, keepdims: bool=False, axis: _1_AllAx=None, out: Optional['Ints1d']=None) -> 'Ints1d':
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Tru, axis: Optional[OneAx]=None, out: Optional['Ints1d']=None) -> 'Ints1d':
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal=False, axis: Optional[OneAx]=None, out=None) -> int:
        ...

    @abstractmethod
    def sum(self, *, keepdims: bool=False, axis: _1_AllAx=None, out: Optional['Ints1d']=None) -> _1I_ReduceResults:
        ...