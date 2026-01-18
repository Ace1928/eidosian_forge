from typing import Callable, List, Optional, Tuple, TypeVar, Union, cast
from ..backends import NumpyOps
from ..config import registry
from ..model import Model
from ..types import Array2d, Floats2d, List2d, Padded, Ragged
def _ragged_forward(model: Model[SeqT, SeqT], Xr: Ragged, is_train: bool) -> Tuple[Ragged, Callable]:
    layer: Model[Array2d, Array2d] = model.layers[0]
    Y, get_dX = layer(Xr.data, is_train)
    x_shape = Xr.dataXd.shape

    def backprop(dYr: Ragged) -> Ragged:
        return Ragged(get_dX(dYr.dataXd).reshape(x_shape), dYr.lengths)
    return (Ragged(Y, Xr.lengths), backprop)