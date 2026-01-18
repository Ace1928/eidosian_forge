from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def _intersection_explicit_with_unfactorized_qid_shapes(explicit_qid_shapes: Set[Tuple[int, ...]], unfactorized_total_dimension: int) -> Set[Tuple[int, ...]]:
    return {qid_shape for qid_shape in explicit_qid_shapes if np.prod(qid_shape, dtype=np.int64) == unfactorized_total_dimension}