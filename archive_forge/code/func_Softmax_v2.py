from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..initializers import zero_init
from ..model import Model
from ..types import Floats1d, Floats2d
from ..util import ArrayInfo, get_width, partial
@registry.layers('Softmax.v2')
def Softmax_v2(nO: Optional[int]=None, nI: Optional[int]=None, *, init_W: Optional[Callable]=None, init_b: Optional[Callable]=None, normalize_outputs: bool=True, temperature: float=1.0) -> Model[InT, OutT]:
    if init_W is None:
        init_W = zero_init
    if init_b is None:
        init_b = zero_init
    validate_temperature(temperature)
    return Model('softmax', forward, init=partial(init, init_W, init_b), dims={'nO': nO, 'nI': nI}, params={'W': None, 'b': None}, attrs={'softmax_normalize': normalize_outputs, 'softmax_temperature': temperature})