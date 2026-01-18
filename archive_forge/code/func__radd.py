from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename
@staticmethod
def _radd(env, x: 'Data', y: 'Data') -> 'Data':
    from cupyx.jit import _internal_types
    if isinstance(x.ctype, Scalar) and x.ctype.dtype.kind in 'iu':
        return _internal_types.Data(f'({x.code} + {y.code})', y.ctype)
    return NotImplemented