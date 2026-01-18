from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..initializers import glorot_uniform_init, zero_init
from ..model import Model
from ..types import Floats1d, Floats2d
from ..util import get_width, partial
@registry.layers('Linear.v1')
def Linear(nO: Optional[int]=None, nI: Optional[int]=None, *, init_W: Optional[Callable]=None, init_b: Optional[Callable]=None) -> Model[InT, OutT]:
    """Multiply inputs by a weights matrix and adds a bias vector."""
    if init_W is None:
        init_W = glorot_uniform_init
    if init_b is None:
        init_b = zero_init
    return Model('linear', forward, init=partial(init, init_W, init_b), dims={'nO': nO, 'nI': nI}, params={'W': None, 'b': None})