from typing import Callable, List, Sequence, Tuple, TypeVar, Union, cast
from ..config import registry
from ..model import Model
from ..types import ArrayXd, Padded, Ragged
def _dropout_array(model: Model[InT, InT], X: ArrayXd, is_train: bool) -> Tuple[ArrayXd, Callable]:
    rate = model.attrs['dropout_rate']
    mask = model.ops.get_dropout_mask(X.shape, rate)

    def backprop(dY: ArrayXd) -> ArrayXd:
        return dY * mask
    return (cast(ArrayXd, X * mask), backprop)