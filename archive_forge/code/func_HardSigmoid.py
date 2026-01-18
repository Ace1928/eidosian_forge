from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..initializers import glorot_uniform_init, zero_init
from ..model import Model
from ..types import Floats1d, Floats2d
from ..util import get_width, partial
from .chain import chain
from .dropout import Dropout
from .layernorm import LayerNorm
@registry.layers('HardSigmoid.v1')
def HardSigmoid(nO: Optional[int]=None, nI: Optional[int]=None, *, init_W: Optional[Callable]=None, init_b: Optional[Callable]=None, dropout: Optional[float]=None, normalize: bool=False) -> Model[Floats2d, Floats2d]:
    if init_W is None:
        init_W = glorot_uniform_init
    if init_b is None:
        init_b = zero_init
    return ClippedLinear(nO=nO, nI=nI, init_W=init_W, dropout=dropout, normalize=normalize, slope=0.2, offset=0.5)