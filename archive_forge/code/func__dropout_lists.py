from typing import Callable, List, Sequence, Tuple, TypeVar, Union, cast
from ..config import registry
from ..model import Model
from ..types import ArrayXd, Padded, Ragged
def _dropout_lists(model: Model[InT, InT], Xs: Sequence[ArrayXd], is_train: bool) -> Tuple[Sequence[ArrayXd], Callable]:
    rate = model.attrs['dropout_rate']
    masks = [model.ops.get_dropout_mask(X.shape, rate) for X in Xs]
    Ys = [X * mask for X, mask in zip(Xs, masks)]

    def backprop(dYs: List[ArrayXd]) -> List[ArrayXd]:
        return [dY * mask for dY, mask in zip(dYs, masks)]
    return (Ys, backprop)