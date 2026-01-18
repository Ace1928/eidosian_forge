from functools import partial
from typing import Callable, Optional, Tuple, cast
from ..backends import Ops
from ..config import registry
from ..initializers import glorot_uniform_init, zero_init
from ..model import Model
from ..types import Floats1d, Floats2d, Floats4d, Padded, Ragged
from ..util import get_width
from .noop import noop
def _padded_to_packed(ops: Ops, Xp: Padded) -> Ragged:
    """Strip padding from a padded sequence."""
    assert Xp.lengths.sum() == Xp.size_at_t.sum(), (Xp.lengths.sum(), Xp.size_at_t.sum())
    Y = ops.alloc2f(Xp.lengths.sum(), Xp.data.shape[2])
    start = 0
    for t in range(Xp.size_at_t.shape[0]):
        batch_size = Xp.size_at_t[t]
        Y[start:start + batch_size] = Xp.data[t, :batch_size]
        start += batch_size
    return Ragged(Y, Xp.size_at_t)