from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..model import Model
from ..types import Floats1d, Floats2d
from ..util import get_width
@registry.layers('CauchySimilarity.v1')
def CauchySimilarity(nI: Optional[int]=None) -> Model[InT, OutT]:
    """Compare input vectors according to the Cauchy similarity function proposed by
    Chen (2013). Primarily used within Siamese neural networks.
    """
    return Model('cauchy_similarity', forward, init=init, dims={'nI': nI, 'nO': 1}, params={'W': None})