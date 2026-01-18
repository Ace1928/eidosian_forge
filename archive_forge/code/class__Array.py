import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
class _Array(Sized, Container):

    @classmethod
    def __get_validators__(cls):
        """Runtime validation for pydantic."""
        yield (lambda v: validate_array(v))

    @property
    @abstractmethod
    def dtype(self) -> DTypes:
        ...

    @property
    @abstractmethod
    def data(self) -> memoryview:
        ...

    @property
    @abstractmethod
    def flags(self) -> Any:
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        ...

    @property
    @abstractmethod
    def itemsize(self) -> int:
        ...

    @property
    @abstractmethod
    def nbytes(self) -> int:
        ...

    @property
    @abstractmethod
    def ndim(self) -> int:
        ...

    @property
    @abstractmethod
    def shape(self) -> Shape:
        ...

    @property
    @abstractmethod
    def strides(self) -> Tuple[int, ...]:
        ...

    @abstractmethod
    def astype(self: ArrayT, dtype: DTypes, order: str=..., casting: str=..., subok: bool=..., copy: bool=...) -> ArrayT:
        ...

    @abstractmethod
    def copy(self: ArrayT, order: str=...) -> ArrayT:
        ...

    @abstractmethod
    def fill(self, value: Any) -> None:
        ...

    @abstractmethod
    def reshape(self: ArrayT, shape: Shape, *, order: str=...) -> ArrayT:
        ...

    @abstractmethod
    def transpose(self: ArrayT, axes: Shape) -> ArrayT:
        ...

    @abstractmethod
    def flatten(self, order: str=...):
        ...

    @abstractmethod
    def ravel(self, order: str=...):
        ...

    @abstractmethod
    def squeeze(self, axis: Union[int, Shape]=...):
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __setitem__(self, key, value):
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        ...

    @abstractmethod
    def __contains__(self, key) -> bool:
        ...

    @abstractmethod
    def __index__(self) -> int:
        ...

    @abstractmethod
    def __int__(self) -> int:
        ...

    @abstractmethod
    def __float__(self) -> float:
        ...

    @abstractmethod
    def __complex__(self) -> complex:
        ...

    @abstractmethod
    def __bool__(self) -> bool:
        ...

    @abstractmethod
    def __bytes__(self) -> bytes:
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...

    @abstractmethod
    def __repr__(self) -> str:
        ...

    @abstractmethod
    def __copy__(self, order: str=...):
        ...

    @abstractmethod
    def __deepcopy__(self: SelfT, memo: dict) -> SelfT:
        ...

    @abstractmethod
    def __lt__(self, other):
        ...

    @abstractmethod
    def __le__(self, other):
        ...

    @abstractmethod
    def __eq__(self, other):
        ...

    @abstractmethod
    def __ne__(self, other):
        ...

    @abstractmethod
    def __gt__(self, other):
        ...

    @abstractmethod
    def __ge__(self, other):
        ...

    @abstractmethod
    def __add__(self, other):
        ...

    @abstractmethod
    def __radd__(self, other):
        ...

    @abstractmethod
    def __iadd__(self, other):
        ...

    @abstractmethod
    def __sub__(self, other):
        ...

    @abstractmethod
    def __rsub__(self, other):
        ...

    @abstractmethod
    def __isub__(self, other):
        ...

    @abstractmethod
    def __mul__(self, other):
        ...

    @abstractmethod
    def __rmul__(self, other):
        ...

    @abstractmethod
    def __imul__(self, other):
        ...

    @abstractmethod
    def __truediv__(self, other):
        ...

    @abstractmethod
    def __rtruediv__(self, other):
        ...

    @abstractmethod
    def __itruediv__(self, other):
        ...

    @abstractmethod
    def __floordiv__(self, other):
        ...

    @abstractmethod
    def __rfloordiv__(self, other):
        ...

    @abstractmethod
    def __ifloordiv__(self, other):
        ...

    @abstractmethod
    def __mod__(self, other):
        ...

    @abstractmethod
    def __rmod__(self, other):
        ...

    @abstractmethod
    def __imod__(self, other):
        ...

    @abstractmethod
    def __divmod__(self, other):
        ...

    @abstractmethod
    def __rdivmod__(self, other):
        ...

    @abstractmethod
    def __pow__(self, other):
        ...

    @abstractmethod
    def __rpow__(self, other):
        ...

    @abstractmethod
    def __ipow__(self, other):
        ...

    @abstractmethod
    def __lshift__(self, other):
        ...

    @abstractmethod
    def __rlshift__(self, other):
        ...

    @abstractmethod
    def __ilshift__(self, other):
        ...

    @abstractmethod
    def __rshift__(self, other):
        ...

    @abstractmethod
    def __rrshift__(self, other):
        ...

    @abstractmethod
    def __irshift__(self, other):
        ...

    @abstractmethod
    def __and__(self, other):
        ...

    @abstractmethod
    def __rand__(self, other):
        ...

    @abstractmethod
    def __iand__(self, other):
        ...

    @abstractmethod
    def __xor__(self, other):
        ...

    @abstractmethod
    def __rxor__(self, other):
        ...

    @abstractmethod
    def __ixor__(self, other):
        ...

    @abstractmethod
    def __or__(self, other):
        ...

    @abstractmethod
    def __ror__(self, other):
        ...

    @abstractmethod
    def __ior__(self, other):
        ...

    @abstractmethod
    def __matmul__(self, other):
        ...

    @abstractmethod
    def __rmatmul__(self, other):
        ...

    @abstractmethod
    def __neg__(self: ArrayT) -> ArrayT:
        ...

    @abstractmethod
    def __pos__(self: ArrayT) -> ArrayT:
        ...

    @abstractmethod
    def __abs__(self: ArrayT) -> ArrayT:
        ...

    @abstractmethod
    def __invert__(self: ArrayT) -> ArrayT:
        ...

    @abstractmethod
    def get(self: ArrayT) -> ArrayT:
        ...

    @abstractmethod
    def all(self, axis: int=-1, out: Optional[ArrayT]=None, keepdims: bool=False) -> ArrayT:
        ...

    @abstractmethod
    def any(self, axis: int=-1, out: Optional[ArrayT]=None, keepdims: bool=False) -> ArrayT:
        ...

    @abstractmethod
    def argmin(self, axis: int=-1, out: Optional[ArrayT]=None) -> ArrayT:
        ...

    @abstractmethod
    def clip(self, a_min: Any, a_max: Any, out: Optional[ArrayT]) -> ArrayT:
        ...

    @abstractmethod
    def max(self, axis: int=-1, out: Optional[ArrayT]=None) -> ArrayT:
        ...

    @abstractmethod
    def min(self, axis: int=-1, out: Optional[ArrayT]=None) -> ArrayT:
        ...

    @abstractmethod
    def nonzero(self: SelfT) -> SelfT:
        ...

    @abstractmethod
    def prod(self, axis: int=-1, dtype: Optional[DTypes]=None, out: Optional[ArrayT]=None, keepdims: bool=False) -> ArrayT:
        ...

    @abstractmethod
    def round(self, decimals: int=0, out: Optional[ArrayT]=None) -> ArrayT:
        ...

    @abstractmethod
    def tobytes(self, order: str='C') -> bytes:
        ...

    @abstractmethod
    def tolist(self) -> List[Any]:
        ...

    @abstractmethod
    def var(self: SelfT, axis: int=-1, dtype: Optional[DTypes]=None, out: Optional[ArrayT]=None, ddof: int=0, keepdims: bool=False) -> SelfT:
        ...