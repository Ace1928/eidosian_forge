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
def _equal_controlled(op1: Controlled, op2: Controlled, **kwargs):
    """Determine whether two Controlled or ControlledOp objects are equal"""
    if [dict(zip(op1.control_wires, op1.control_values)), op1.work_wires, op1.arithmetic_depth] != [dict(zip(op2.control_wires, op2.control_values)), op2.work_wires, op2.arithmetic_depth]:
        return False
    return qml.equal(op1.base, op2.base, **kwargs)