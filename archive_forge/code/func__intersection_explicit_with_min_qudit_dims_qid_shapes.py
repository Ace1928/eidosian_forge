from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def _intersection_explicit_with_min_qudit_dims_qid_shapes(explicit_qid_shapes: Set[Tuple[int, ...]], min_qudit_dimensions: Tuple[int, ...]) -> Set[Tuple[int, ...]]:
    return {qid_shape for qid_shape in explicit_qid_shapes if len(qid_shape) == len(min_qudit_dimensions) and all((dim1 >= dim2 for dim1, dim2 in zip(qid_shape, min_qudit_dimensions)))}