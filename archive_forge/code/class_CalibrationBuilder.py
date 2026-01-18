from abc import abstractmethod
from typing import List, Union
from qiskit.circuit import Instruction as CircuitInst
from qiskit.dagcircuit import DAGCircuit
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.calibration_entries import CalibrationPublisher
from qiskit.transpiler.basepasses import TransformationPass
from .exceptions import CalibrationNotAvailable
class CalibrationBuilder(TransformationPass):
    """Abstract base class to inject calibrations into circuits."""

    @abstractmethod
    def supported(self, node_op: CircuitInst, qubits: List) -> bool:
        """Determine if a given node supports the calibration.

        Args:
            node_op: Target instruction object.
            qubits: Integer qubit indices to check.

        Returns:
            Return ``True`` is calibration can be provided.
        """

    @abstractmethod
    def get_calibration(self, node_op: CircuitInst, qubits: List) -> Union[Schedule, ScheduleBlock]:
        """Gets the calibrated schedule for the given instruction and qubits.

        Args:
            node_op: Target instruction object.
            qubits: Integer qubit indices to check.

        Returns:
            Return Schedule of target gate instruction.
        """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the calibration adder pass on `dag`.

        Args:
            dag: DAG to schedule.

        Returns:
            A DAG with calibrations added to it.
        """
        for node in dag.gate_nodes():
            qubits = [dag.find_bit(q).index for q in node.qargs]
            if self.supported(node.op, qubits) and (not dag.has_calibration_for(node)):
                try:
                    schedule = self.get_calibration(node.op, qubits)
                except CalibrationNotAvailable:
                    continue
                publisher = schedule.metadata.get('publisher', CalibrationPublisher.QISKIT)
                if publisher != CalibrationPublisher.BACKEND_PROVIDER:
                    dag.add_calibration(gate=node.op, qubits=qubits, schedule=schedule)
        return dag