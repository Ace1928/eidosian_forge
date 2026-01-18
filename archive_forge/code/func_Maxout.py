from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..initializers import glorot_uniform_init, zero_init
from ..model import Model
from ..types import Floats2d
from ..util import get_width, partial
from .chain import chain
from .dropout import Dropout
from .layernorm import LayerNorm
@registry.layers('Maxout.v1')
def Maxout(nO: Optional[int]=None, nI: Optional[int]=None, nP: Optional[int]=3, *, init_W: Optional[Callable]=None, init_b: Optional[Callable]=None, dropout: Optional[float]=None, normalize: bool=False) -> Model[InT, OutT]:
    if init_W is None:
        init_W = glorot_uniform_init
    if init_b is None:
        init_b = zero_init
    model: Model[InT, OutT] = Model('maxout', forward, init=partial(init, init_W, init_b), dims={'nO': nO, 'nI': nI, 'nP': nP}, params={'W': None, 'b': None})
    if normalize:
        model = chain(model, LayerNorm(nI=nO))
    if dropout is not None:
        model = chain(model, cast(Model[InT, OutT], Dropout(dropout)))
    return model