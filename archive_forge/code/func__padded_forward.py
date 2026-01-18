from typing import Callable, List, Optional, Tuple, TypeVar, Union, cast
from ..backends import NumpyOps
from ..config import registry
from ..model import Model
from ..types import Array2d, Ints1d, List2d, ListXd, Padded, Ragged
def _padded_forward(layer: Model[Ragged, Ragged], Xp: Padded, is_train: bool) -> Tuple[Padded, Callable]:
    list2padded = layer.ops.list2padded
    padded2list = layer.ops.padded2list
    unflatten = layer.ops.unflatten
    flatten = layer.ops.flatten
    Xs = padded2list(Xp)
    lengths = NUMPY_OPS.asarray1i([len(x) for x in Xs])
    Yr, get_dXr = layer(Ragged(flatten(Xs), layer.ops.asarray1i(lengths)), is_train)

    def backprop(dYp: Padded):
        flattened = flatten(padded2list(dYp))
        dXr = get_dXr(Ragged(flattened, lengths))
        return list2padded(unflatten(dXr.data, lengths))
    return (list2padded(unflatten(Yr.data, Yr.lengths)), backprop)