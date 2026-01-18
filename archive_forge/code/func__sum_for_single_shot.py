from typing import List, Union, Tuple
import numpy as np
import pennylane as qml
from pennylane.ops import Sum, Hamiltonian, SProd, Prod
from pennylane.measurements import (
from pennylane.typing import TensorLike
from .apply_operation import apply_operation
from .measure import flatten_state
def _sum_for_single_shot(s):
    results = measure_with_samples([ExpectationMP(t) for t in mp.obs], state, s, is_state_batched=is_state_batched, rng=rng, prng_key=prng_key)
    return sum(results)