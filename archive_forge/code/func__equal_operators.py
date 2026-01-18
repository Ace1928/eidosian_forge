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
def _equal_operators(op1: Operator, op2: Operator, check_interface=True, check_trainability=True, rtol=1e-05, atol=1e-09):
    """Default function to determine whether two Operator objects are equal."""
    if not isinstance(op2, type(op1)):
        return False
    if op1.arithmetic_depth != op2.arithmetic_depth:
        return False
    if op1.arithmetic_depth > 0:
        return False
    if not all((qml.math.allclose(d1, d2, rtol=rtol, atol=atol) for d1, d2 in zip(op1.data, op2.data))):
        return False
    if op1.wires != op2.wires:
        return False
    if op1.hyperparameters != op2.hyperparameters:
        return False
    if check_trainability:
        for params_1, params_2 in zip(op1.data, op2.data):
            if qml.math.requires_grad(params_1) != qml.math.requires_grad(params_2):
                return False
    if check_interface:
        for params_1, params_2 in zip(op1.data, op2.data):
            if qml.math.get_interface(params_1) != qml.math.get_interface(params_2):
                return False
    return True