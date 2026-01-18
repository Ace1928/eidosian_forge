from typing import Callable, Optional, Tuple
from ..config import registry
from ..model import Model
from ..types import Ragged
from ..util import get_width
@registry.layers('ParametricAttention.v1')
def ParametricAttention(nO: Optional[int]=None) -> Model[InT, OutT]:
    """Weight inputs by similarity to a learned vector"""
    return Model('para-attn', forward, init=init, params={'Q': None}, dims={'nO': nO})