from typing import Callable, List, Sequence, Tuple, TypeVar, Union, cast
from ..config import registry
from ..model import Model
from ..types import ArrayXd, Padded, Ragged
def _dropout_ragged(model: Model[InT, InT], Xr: Ragged, is_train: bool) -> Tuple[Ragged, Callable]:
    X = Xr.data
    lengths = Xr.lengths
    mask = model.ops.get_dropout_mask(X.shape, model.attrs['dropout_rate'])
    Y = X * mask

    def backprop(dYr: Ragged) -> Ragged:
        return Ragged(dYr.data * mask, dYr.lengths)
    return (Ragged(Y, lengths), backprop)