from functools import partial
from typing import Callable, Optional, Tuple, cast
from ..backends import Ops
from ..config import registry
from ..initializers import glorot_uniform_init, zero_init
from ..model import Model
from ..types import Floats1d, Floats2d, Floats4d, Padded, Ragged
from ..util import get_width
from .noop import noop
@registry.layers('LSTM.v1')
def LSTM(nO: Optional[int]=None, nI: Optional[int]=None, *, bi: bool=False, depth: int=1, dropout: float=0.0, init_W: Optional[Callable]=None, init_b: Optional[Callable]=None) -> Model[Padded, Padded]:
    if depth == 0:
        msg = 'LSTM depth must be at least 1. Maybe we should make this a noop?'
        raise ValueError(msg)
    if init_W is None:
        init_W = glorot_uniform_init
    if init_b is None:
        init_b = zero_init
    model: Model[Padded, Padded] = Model('lstm', forward, dims={'nO': nO, 'nI': nI, 'depth': depth, 'dirs': 1 + int(bi)}, attrs={'registry_name': 'LSTM.v1', 'dropout_rate': dropout}, params={'LSTM': None, 'HC0': None}, init=partial(init, init_W, init_b))
    return model