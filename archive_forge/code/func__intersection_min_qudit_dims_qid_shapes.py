from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def _intersection_min_qudit_dims_qid_shapes(min_qudit_dimensions1: Tuple[int, ...], min_qudit_dimensions2: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    if len(min_qudit_dimensions1) == len(min_qudit_dimensions2):
        return tuple((max(dim1, dim2) for dim1, dim2 in zip(min_qudit_dimensions1, min_qudit_dimensions2)))
    return None