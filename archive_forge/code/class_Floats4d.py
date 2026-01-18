import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
class Floats4d(_Array4d, _Floats):
    """4-dimensional array of floats."""
    T: 'Floats4d'

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield (lambda v: validate_array(v, ndim=4, dtype='f'))

    @abstractmethod
    def __iter__(self) -> Iterator[Floats3d]:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _4_KeyScalar) -> float:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _4_Key1d) -> Floats1d:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _4_Key2d) -> Floats2d:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _4_Key3d) -> Floats3d:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _4_Key4d) -> 'Floats4d':
        ...

    @abstractmethod
    def __getitem__(self, key: _4_AllKeys) -> _F4_AllReturns:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _4_KeyScalar, value: float) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _4_Key1d, value: Floats1d) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _4_Key2d, value: Floats2d) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _4_Key3d, value: Floats3d) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _4_Key4d, value: 'Floats4d') -> None:
        ...

    @abstractmethod
    def __setitem__(self, key: _4_AllKeys, value: _F4_AllReturns) -> None:
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Tru, axis: _4_AllAx=None, out: Optional['Floats4d']=None) -> 'Floats4d':
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal=False, axis: OneAx, out: Optional[Floats3d]=None) -> Floats3d:
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal=False, axis: TwoAx, out: Optional[Floats2d]=None) -> Floats2d:
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal=False, axis: ThreeAx, out: Optional[Floats1d]=None) -> Floats1d:
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal=False, axis: Optional[FourAx], out=None) -> float:
        ...

    @abstractmethod
    def sum(self, *, keepdims: bool=False, axis: _4_AllAx=None, out: Union[None, Floats1d, Floats2d, Floats3d, 'Floats4d']=None) -> _4F_ReduceResults:
        ...