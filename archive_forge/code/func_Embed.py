from typing import Callable, Dict, Optional, Tuple, TypeVar, Union, cast
from ..config import registry
from ..initializers import uniform_init
from ..model import Model
from ..types import Floats1d, Floats2d, Ints1d, Ints2d
from ..util import get_width, partial
from .array_getitem import ints_getitem
from .chain import chain
@registry.layers('Embed.v1')
def Embed(nO: Optional[int]=None, nV: Optional[int]=None, *, column: Optional[int]=None, initializer: Optional[Callable]=None, dropout: Optional[float]=None) -> Model[InT, OutT]:
    """Map integers to vectors, using a fixed-size lookup table."""
    attrs: Dict[str, Union[None, int, float]] = {}
    if initializer is None:
        initializer = uniform_init
    if dropout is not None:
        attrs['dropout_rate'] = dropout
    model: Model = Model('embed', forward, init=partial(init, initializer), attrs=attrs, dims={'nO': nO, 'nV': nV}, params={'E': None})
    if column is not None:
        model = chain(ints_getitem((slice(0, None), column)), model)
    model.attrs['column'] = column
    return cast(Model[InT, OutT], model)