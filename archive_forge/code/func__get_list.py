from typing import Callable, List, Optional, Tuple, TypeVar, Union, cast
from ..config import registry
from ..model import Model
from ..types import Array2d, Floats2d, Ints2d, List2d, Padded, Ragged
def _get_list(model, seq):
    if isinstance(seq, Padded):
        return model.ops.padded2list(seq)
    elif isinstance(seq, Ragged):
        return model.ops.unflatten(seq.data, seq.lengths)
    else:
        return seq