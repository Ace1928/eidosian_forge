import numpy as np
import pennylane as qml
from pennylane.measurements import ProbabilityMP
def _single_measurement_zero(m, tangent):
    """Aux function to create a zero tensor from a measurement."""
    dim = 2 ** len(m.wires) if isinstance(m, ProbabilityMP) else ()
    res = qml.math.convert_like(np.zeros(dim), tangent)
    res = qml.math.cast_like(res, tangent)
    return res