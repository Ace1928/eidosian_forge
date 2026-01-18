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
def _equal_basis_rotation(op1: qml.BasisRotation, op2: qml.BasisRotation, check_interface=True, check_trainability=True, rtol=1e-05, atol=1e-09):
    if not qml.math.allclose(op1.hyperparameters['unitary_matrix'], op2.hyperparameters['unitary_matrix'], atol=atol, rtol=rtol):
        return False
    if op1.wires != op2.wires:
        return False
    if check_interface:
        if qml.math.get_interface(op1.hyperparameters['unitary_matrix']) != qml.math.get_interface(op2.hyperparameters['unitary_matrix']):
            return False
    return True