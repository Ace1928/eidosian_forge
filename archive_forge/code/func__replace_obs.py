import warnings
from functools import reduce, partial
from itertools import product
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.measurements import ClassicalShadowMP
from pennylane.tape import QuantumScript, QuantumTape
from pennylane import transform
@transform
def _replace_obs(tape: QuantumTape, obs, *args, **kwargs) -> (Sequence[QuantumTape], Callable):
    """
    Tape transform to replace the measurement processes with the given one
    """
    for m in tape.measurements:
        if not isinstance(m, ClassicalShadowMP):
            raise ValueError(f'Tape measurement must be ClassicalShadowMP, got {m.__class__.__name__!r}')
    with qml.queuing.AnnotatedQueue() as q:
        for op in tape.operations:
            qml.apply(op)
        obs(*args, **kwargs)
    qscript = QuantumScript.from_queue(q, shots=tape.shots)

    def processing_fn(res):
        return res[0]
    return ([qscript], processing_fn)