from typing import Callable, List, Optional, Tuple, TypeVar, Union, cast
from ..backends import NumpyOps
from ..config import registry
from ..model import Model
from ..types import Array2d, Ints1d, List2d, ListXd, Padded, Ragged
def _list_forward(layer: Model[Ragged, Ragged], Xs: List, is_train: bool) -> Tuple[List, Callable]:
    flatten = layer.ops.flatten
    unflatten = layer.ops.unflatten
    lengths = [len(x) for x in Xs]
    Yr, get_dXr = layer(Ragged(flatten(Xs), layer.ops.asarray1i(lengths)), is_train)

    def backprop(dYs):
        flattened = flatten(dYs)
        return unflatten(get_dXr(Ragged(flattened, lengths)).data, lengths)
    return (unflatten(Yr.data, Yr.lengths), backprop)