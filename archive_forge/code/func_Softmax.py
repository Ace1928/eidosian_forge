from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..initializers import zero_init
from ..model import Model
from ..types import Floats1d, Floats2d
from ..util import ArrayInfo, get_width, partial
@registry.layers('Softmax.v1')
def Softmax(nO: Optional[int]=None, nI: Optional[int]=None, *, init_W: Optional[Callable]=None, init_b: Optional[Callable]=None) -> Model[InT, OutT]:
    if init_W is None:
        init_W = zero_init
    if init_b is None:
        init_b = zero_init
    return Model('softmax', forward, init=partial(init, init_W, init_b), dims={'nO': nO, 'nI': nI}, params={'W': None, 'b': None}, attrs={'softmax_normalize': True, 'softmax_temperature': 1.0})