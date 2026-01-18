from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
def _assert_same_pointer_type(a: _Data, b: _Data) -> None:
    _assert_pointer_type(a)
    _assert_pointer_type(b)
    if a.ctype.child_type != b.ctype.child_type:
        raise TypeError(f'`{a.code}` and `{b.code}` must be of the same pointer type: `{a.ctype.child_type}` != `{b.type.child_type}`')