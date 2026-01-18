from typing import Callable, Tuple, TypeVar, Union, cast
from ..config import registry
from ..model import Model
from ..types import Floats2d, Ragged
def _expand_window_floats(model: Model[InT, InT], X: Floats2d) -> Tuple[Floats2d, Callable]:
    nW = model.attrs['window_size']
    if len(X) > 0:
        Y = model.ops.seq2col(X, nW)
    else:
        assert len(X) == 0
        Y = model.ops.tile(X, nW * 2 + 1)

    def backprop(dY: Floats2d) -> Floats2d:
        return model.ops.backprop_seq2col(dY, nW)
    return (Y, backprop)