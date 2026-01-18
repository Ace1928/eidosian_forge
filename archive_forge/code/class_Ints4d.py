import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
class Ints4d(_Array4d, _Ints):
    """4-dimensional array of ints."""
    T: 'Ints4d'

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield (lambda v: validate_array(v, ndim=4, dtype='i'))

    @abstractmethod
    def __iter__(self) -> Iterator[Ints3d]:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _4_KeyScalar) -> int:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _4_Key1d) -> Ints1d:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _4_Key2d) -> Ints2d:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _4_Key3d) -> Ints3d:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _4_Key4d) -> 'Ints4d':
        ...

    @abstractmethod
    def __getitem__(self, key: _4_AllKeys) -> _I4_AllReturns:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _4_KeyScalar, value: int) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _4_Key1d, value: Ints1d) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _4_Key2d, value: Ints2d) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _4_Key3d, value: Ints3d) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _4_Key4d, value: 'Ints4d') -> None:
        ...

    @abstractmethod
    def __setitem__(self, key: _4_AllKeys, value: _I4_AllReturns) -> None:
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Tru, axis: _4_AllAx=None, out: Optional['Ints4d']=None) -> 'Ints4d':
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal=False, axis: OneAx, out: Optional[Ints3d]=None) -> Ints3d:
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal=False, axis: TwoAx, out: Optional[Ints2d]=None) -> Ints2d:
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal=False, axis: ThreeAx, out: Optional[Ints1d]=None) -> Ints1d:
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal=False, axis: Optional[FourAx]=None, out=None) -> int:
        ...

    @abstractmethod
    def sum(self, *, keepdims: bool=False, axis: _4_AllAx=None, out: Optional[Union[Ints1d, Ints2d, Ints3d, 'Ints4d']]=None) -> _4I_ReduceResults:
        ...