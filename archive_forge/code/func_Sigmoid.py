from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..initializers import zero_init
from ..model import Model
from ..types import Floats1d, Floats2d
from ..util import get_width, partial
@registry.layers('Sigmoid.v1')
def Sigmoid(nO: Optional[int]=None, nI: Optional[int]=None, *, init_W: Optional[Callable]=None, init_b: Optional[Callable]=None) -> Model[InT, OutT]:
    """A dense layer, followed by a sigmoid (logistic) activation function. This
    is usually used instead of the Softmax layer as an output for multi-label
    classification.
    """
    if init_W is None:
        init_W = zero_init
    if init_b is None:
        init_b = zero_init
    return Model('sigmoid', forward, init=partial(init, init_W, init_b), dims={'nO': nO, 'nI': nI}, params={'W': None, 'b': None})