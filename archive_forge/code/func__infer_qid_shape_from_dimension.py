from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def _infer_qid_shape_from_dimension(dim: int) -> Tuple[int, ...]:
    if dim != 0 and dim & dim - 1 == 0:
        n_qubits = dim.bit_length() - 1
        return (2,) * n_qubits
    return (dim,)