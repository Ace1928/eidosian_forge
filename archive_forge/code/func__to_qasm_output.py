import abc
import enum
import html
import itertools
import math
from collections import defaultdict
from typing import (
from typing_extensions import Self
import networkx
import numpy as np
import cirq._version
from cirq import _compat, devices, ops, protocols, qis
from cirq._doc import document
from cirq.circuits._bucket_priority_queue import BucketPriorityQueue
from cirq.circuits.circuit_operation import CircuitOperation
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.qasm_output import QasmOutput
from cirq.circuits.text_diagram_drawer import TextDiagramDrawer
from cirq.circuits.moment import Moment
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def _to_qasm_output(self, header: Optional[str]=None, precision: int=10, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> 'cirq.QasmOutput':
    """Returns a QASM object equivalent to the circuit.

        Args:
            header: A multi-line string that is placed in a comment at the top
                of the QASM. Defaults to a cirq version specifier.
            precision: Number of digits to use when representing numbers.
            qubit_order: Determines how qubits are ordered in the QASM
                register.
        """
    if header is None:
        header = f'Generated from Cirq v{cirq._version.__version__}'
    qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(self.all_qubits())
    return QasmOutput(operations=self.all_operations(), qubits=qubits, header=header, precision=precision, version='2.0')