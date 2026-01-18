from collections.abc import Iterable
from functools import singledispatch
from typing import Union
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.measurements.classical_shadow import ShadowExpvalMP
from pennylane.measurements.mid_measure import MidMeasureMP, MeasurementValue
from pennylane.measurements.mutual_info import MutualInfoMP
from pennylane.measurements.vn_entropy import VnEntropyMP
from pennylane.measurements.counts import CountsMP
from pennylane.pulse.parametrized_evolution import ParametrizedEvolution
from pennylane.operation import Observable, Operator, Tensor
from pennylane.ops import Hamiltonian, Controlled, Pow, Adjoint, Exp, SProd, CompositeOp
from pennylane.templates.subroutines import ControlledSequence
from pennylane.tape import QuantumTape
@_equal.register
def _equal_parametrized_evolution(op1: ParametrizedEvolution, op2: ParametrizedEvolution, **kwargs):
    if op1.t is None or op2.t is None:
        if not (op1.t is None and op2.t is None):
            return False
    elif not qml.math.allclose(op1.t, op2.t):
        return False
    if not _equal_operators(op1, op2, **kwargs):
        return False
    if not all((c1 == c2 for c1, c2 in zip(op1.H.coeffs, op2.H.coeffs))):
        return False
    return all((equal(o1, o2, **kwargs) for o1, o2 in zip(op1.H.ops, op2.H.ops)))