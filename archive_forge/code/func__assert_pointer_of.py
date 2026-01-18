from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
def _assert_pointer_of(a: _Data, b: _Data) -> None:
    _assert_pointer_type(a)
    if a.ctype.child_type != b.ctype:
        raise TypeError(f'`*{a.code}` and `{b.code}` must be of the same type: `{a.ctype.child_type}` != `{b.ctype}`')