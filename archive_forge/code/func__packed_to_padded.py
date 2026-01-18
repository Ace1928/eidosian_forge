from functools import partial
from typing import Callable, Optional, Tuple, cast
from ..backends import Ops
from ..config import registry
from ..initializers import glorot_uniform_init, zero_init
from ..model import Model
from ..types import Floats1d, Floats2d, Floats4d, Padded, Ragged
from ..util import get_width
from .noop import noop
def _packed_to_padded(ops: Ops, Xr: Ragged, Xp: Padded) -> Padded:
    Y = ops.alloc3f(Xp.data.shape[0], Xp.data.shape[1], Xr.data.shape[1])
    X = cast(Floats2d, Xr.data)
    start = 0
    for t in range(Xp.size_at_t.shape[0]):
        batch_size = Xp.size_at_t[t]
        Y[t, :batch_size] = X[start:start + batch_size]
        start += batch_size
    return Padded(Y, size_at_t=Xp.size_at_t, lengths=Xp.lengths, indices=Xp.indices)