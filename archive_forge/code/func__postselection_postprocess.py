from collections import Counter
from typing import Optional, Sequence
import warnings
from numpy.random import default_rng
import numpy as np
import pennylane as qml
from pennylane.measurements import (
from pennylane.typing import Result
from .initialize_state import create_initial_state
from .apply_operation import apply_operation
from .measure import measure
from .sampling import measure_with_samples
def _postselection_postprocess(state, is_state_batched, shots):
    """Update state after projector is applied."""
    if is_state_batched:
        raise ValueError('Cannot postselect on circuits with broadcasting. Use the qml.transforms.broadcast_expand transform to split a broadcasted tape into multiple non-broadcasted tapes before executing if postselection is used.')
    norm = qml.math.norm(state)
    if not qml.math.is_abstract(state) and qml.math.allclose(norm, 0.0):
        norm = 0.0
    if shots:
        postselected_shots = [np.random.binomial(s, float(norm ** 2)) for s in shots] if not qml.math.is_abstract(norm) else shots
        shots = _FlexShots(postselected_shots)
    state = state / norm
    return (state, shots)