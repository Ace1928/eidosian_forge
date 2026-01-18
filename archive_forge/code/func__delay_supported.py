from qiskit.transpiler import InstructionDurations
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.scheduling.time_unit_conversion import TimeUnitConversion
from qiskit.dagcircuit import DAGOpNode, DAGCircuit
from qiskit.circuit import Delay, Gate
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target
def _delay_supported(self, qarg: int) -> bool:
    """Delay operation is supported on the qubit (qarg) or not."""
    if self.target is None or self.target.instruction_supported('delay', qargs=(qarg,)):
        return True
    return False