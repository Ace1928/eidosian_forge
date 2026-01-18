from typing import Optional, TYPE_CHECKING, Tuple
import numpy as np
from cirq import protocols, value, _import
from cirq.qis.states import (
def _validate_product_state(state: 'cirq.ProductState', qid_shape: Optional[Tuple[int, ...]]) -> None:
    if qid_shape is not None and qid_shape != (2,) * len(state):
        raise ValueError(f'Invalid state for given qid shape: Specified shape {qid_shape} but product state has shape {(2,) * len(state)}.')