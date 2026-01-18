import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
class Ints3d(_Array3d, _Ints):
    """3-dimensional array of ints."""
    T: 'Ints3d'

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield (lambda v: validate_array(v, ndim=3, dtype='i'))

    @abstractmethod
    def __iter__(self) -> Iterator[Ints2d]:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _3_KeyScalar) -> int:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _3_Key1d) -> Ints1d:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _3_Key2d) -> Ints2d:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, key: _3_Key3d) -> 'Ints3d':
        ...

    @abstractmethod
    def __getitem__(self, key: _3_AllKeys) -> _I3_AllReturns:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _3_KeyScalar, value: int) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _3_Key1d, value: Ints1d) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _3_Key2d, value: Ints2d) -> None:
        ...

    @overload
    @abstractmethod
    def __setitem__(self, key: _3_Key3d, value: 'Ints3d') -> None:
        ...

    @abstractmethod
    def __setitem__(self, key: _3_AllKeys, value: _I3_AllReturns) -> None:
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Tru, axis: _3_AllAx=None, out: Optional['Ints3d']=None) -> 'Ints3d':
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal, axis: OneAx, out: Optional[Ints2d]=None) -> Ints2d:
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal, axis: TwoAx, out: Optional[Ints1d]=None) -> Ints1d:
        ...

    @overload
    @abstractmethod
    def sum(self, *, keepdims: Fal, axis: Optional[ThreeAx], out=None) -> int:
        ...

    @abstractmethod
    def sum(self, *, keepdims: bool=False, axis: _3_AllAx=None, out: Union[None, Ints1d, Ints2d, 'Ints3d']=None) -> _3I_ReduceResults:
        ...