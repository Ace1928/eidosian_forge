from abc import abstractmethod
from typing import List, Union
from qiskit.circuit import Instruction as CircuitInst
from qiskit.dagcircuit import DAGCircuit
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.calibration_entries import CalibrationPublisher
from qiskit.transpiler.basepasses import TransformationPass
from .exceptions import CalibrationNotAvailable
Run the calibration adder pass on `dag`.

        Args:
            dag: DAG to schedule.

        Returns:
            A DAG with calibrations added to it.
        