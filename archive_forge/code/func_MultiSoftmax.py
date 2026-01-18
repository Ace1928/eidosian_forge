from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..model import Model
from ..types import Floats1d, Floats2d
from ..util import get_width
@registry.layers('MultiSoftmax.v1')
def MultiSoftmax(nOs: Tuple[int, ...], nI: Optional[int]=None) -> Model[InT, OutT]:
    """Neural network layer that predicts several multi-class attributes at once.
    For instance, we might predict one class with 6 variables, and another with 5.
    We predict the 11 neurons required for this, and then softmax them such
    that columns 0-6 make a probability distribution and columns 6-11 make another.
    """
    return Model('multisoftmax', forward, init=init, dims={'nO': sum(nOs), 'nI': nI}, attrs={'nOs': nOs}, params={'W': None, 'b': None})