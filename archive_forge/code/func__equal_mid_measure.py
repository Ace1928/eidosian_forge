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
def _equal_mid_measure(op1: MidMeasureMP, op2: MidMeasureMP, **_):
    return op1.wires == op2.wires and op1.id == op2.id and (op1.reset == op2.reset) and (op1.postselect == op2.postselect)