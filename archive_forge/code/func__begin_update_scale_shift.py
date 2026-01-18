from typing import Callable, Optional, Tuple, cast
from ..backends import Ops
from ..config import registry
from ..model import Model
from ..types import Floats2d
from ..util import get_width
def _begin_update_scale_shift(model: Model[InT, InT], X: InT) -> Tuple[InT, Callable]:
    G = model.get_param('G')
    b = model.get_param('b')
    Y = X * G
    Y += b

    def finish_update_scale_shift(dY: InT) -> InT:
        model.inc_grad('b', dY.sum(axis=0))
        model.inc_grad('G', (dY * X).sum(axis=0))
        return dY * G
    return (Y, finish_update_scale_shift)