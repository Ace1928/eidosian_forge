from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def _qudit_values_to_state_tensor(*, state_vector: np.ndarray, qid_shape: Tuple[int, ...], dtype: Optional['DTypeLike']) -> np.ndarray:
    for i in range(len(qid_shape)):
        s = state_vector[i]
        q = qid_shape[i]
        if not 0 <= s < q:
            raise ValueError(f'Qudit value {s} at index {i} is out of bounds for qudit dimension {q}.\n\nqid_shape={qid_shape!r}\nstate={state_vector!r}\n')
    if state_vector.dtype.kind[0] not in '?bBiu':
        raise ValueError(f'Expected a bool or int entry for each qudit in `state`, because len(state) == len(qid_shape), but got dtype {state_vector.dtype}.\nqid_shape={qid_shape!r}\nstate={state_vector!r}\n')
    if dtype is None:
        dtype = DEFAULT_COMPLEX_DTYPE
    return one_hot(index=tuple((int(e) for e in state_vector)), shape=qid_shape, dtype=dtype)