from typing import Callable, Tuple, TypeVar, Union, cast
from ..config import registry
from ..model import Model
from ..types import Floats2d, Ragged
def _expand_window_ragged(model: Model[InT, InT], Xr: Ragged) -> Tuple[Ragged, Callable]:
    nW = model.attrs['window_size']
    Y = Ragged(model.ops.seq2col(cast(Floats2d, Xr.data), nW, lengths=Xr.lengths), Xr.lengths)

    def backprop(dYr: Ragged) -> Ragged:
        return Ragged(model.ops.backprop_seq2col(cast(Floats2d, dYr.data), nW, lengths=Xr.lengths), Xr.lengths)
    return (Y, backprop)