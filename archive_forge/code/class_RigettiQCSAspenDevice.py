from typing import List, cast, Optional, Union, Dict, Any
import functools
from math import sqrt
import httpx
import numpy as np
import networkx as nx
import cirq
from pyquil.quantum_processor import QCSQuantumProcessor
from qcs_api_client.models import InstructionSetArchitecture
from qcs_api_client.operations.sync import get_instruction_set_architecture
from cirq_rigetti._qcs_api_client_decorator import _provide_default_client
@cirq.value.value_equality
class RigettiQCSAspenDevice(cirq.devices.Device):
    """A cirq.Qid supporting Rigetti QCS Aspen device topology."""

    def __init__(self, isa: Union[InstructionSetArchitecture, Dict[str, Any]]) -> None:
        """Initializes a RigettiQCSAspenDevice with its Rigetti QCS `InstructionSetArchitecture`.

        Args:
            isa: The `InstructionSetArchitecture` retrieved from the QCS api.

        Raises:
            UnsupportedRigettiQCSQuantumProcessor: If the isa does not define
            an Aspen device.
        """
        if isinstance(isa, InstructionSetArchitecture):
            self.isa = isa
        else:
            self.isa = InstructionSetArchitecture.from_dict(isa)
        if self.isa.architecture.family.lower() != 'aspen':
            raise UnsupportedRigettiQCSQuantumProcessor(f'this integration currently only supports Aspen devices, but client provided a {self.isa.architecture.family} device')
        self.quantum_processor = QCSQuantumProcessor(quantum_processor_id=self.isa.name, isa=self.isa)

    def qubits(self) -> List['AspenQubit']:
        """Return list of `AspenQubit`s within device topology.

        Returns:
            List of `AspenQubit`s within device topology.
        """
        qubits = []
        for node in self.isa.architecture.nodes:
            qubits.append(AspenQubit.from_aspen_index(node.node_id))
        return qubits

    @property
    def qubit_topology(self) -> nx.Graph:
        """Return qubit topology indices with nx.Graph.

        Returns:
            Qubit topology as nx.Graph with each node specified with AspenQubit index.
        """
        return self.quantum_processor.qubit_topology()

    @property
    def _number_octagons(self) -> int:
        return int(np.ceil(self._maximum_qubit_number / 10))

    @property
    def _maximum_qubit_number(self) -> int:
        return max([node.node_id for node in self.isa.architecture.nodes])

    @functools.lru_cache(maxsize=2)
    def _line_qubit_mapping(self) -> List[int]:
        mapping: List[int] = []
        for i in range(self._number_octagons):
            base = i * 10
            mapping = mapping + [base + index for index in _forward_line_qubit_mapping]
        for i in range(self._number_octagons):
            base = (self._number_octagons - i - 1) * 10
            mapping = mapping + [base + index for index in _reverse_line_qubit_mapping]
        return mapping

    def _aspen_qubit_index(self, valid_qubit: cirq.Qid) -> int:
        if isinstance(valid_qubit, cirq.GridQubit):
            return _grid_qubit_mapping[valid_qubit]
        if isinstance(valid_qubit, cirq.LineQubit):
            return self._line_qubit_mapping()[valid_qubit.x]
        if isinstance(valid_qubit, cirq.NamedQubit):
            return int(valid_qubit.name)
        if isinstance(valid_qubit, (OctagonalQubit, AspenQubit)):
            return valid_qubit.index
        else:
            raise UnsupportedQubit(f'unsupported Qid type {type(valid_qubit)}')

    def validate_qubit(self, qubit: 'cirq.Qid') -> None:
        """Raises an exception if the qubit does not satisfy the topological constraints
        of the RigettiQCSAspenDevice.

        Args:
            qubit: The qubit to validate.

        Raises:
            UnsupportedQubit: The operation isn't valid for this device.
        """
        if isinstance(qubit, cirq.GridQubit):
            if self._number_octagons < 2:
                raise UnsupportedQubit('this device does not support GridQubits')
            if not (qubit.row <= 1 and qubit.col <= 1):
                raise UnsupportedQubit('Aspen devices only support square grids of 1 row and 1 column')
            return
        if isinstance(qubit, cirq.LineQubit):
            if not qubit.x <= self._number_octagons * 8:
                raise UnsupportedQubit('this Aspen device only supports line ', f'qubits up to length {self._number_octagons * 8}')
            return
        if isinstance(qubit, cirq.NamedQubit):
            try:
                index = int(qubit.name)
                if not index < self._maximum_qubit_number:
                    raise UnsupportedQubit(f'this Aspen device only supports qubits up to index {self._maximum_qubit_number}')
                if not index % 10 <= 7:
                    raise UnsupportedQubit('this Aspen device only supports qubit indices mod 10 <= 7')
                return
            except ValueError:
                raise UnsupportedQubit('Aspen devices only support named qubits by octagonal index')
        if isinstance(qubit, (OctagonalQubit, AspenQubit)):
            if not qubit.index < self._maximum_qubit_number:
                raise UnsupportedQubit('this Aspen device only supports ', f'qubits up to index {self._maximum_qubit_number}')
            return
        else:
            raise UnsupportedQubit(f'unsupported Qid type {type(qubit)}')

    def validate_operation(self, operation: 'cirq.Operation') -> None:
        """Raises an exception if an operation does not satisfy the topological constraints
        of the device.

        Note, in case the operation is invalid, you can still use the Quil
        compiler to rewire qubits and decompose the operation to this device's
        topology.

        Additionally, this method will not attempt to decompose the operation into this
        device's native gate set. This integration, by default, uses the Quil
        compiler to do so.

        Please see the Quil Compiler
        [documentation](https://pyquil-docs.rigetti.com/en/stable/compiler.html)
        for more information.

        Args:
            operation: The operation to validate.

        Raises:
            UnsupportedRigettiQCSOperation: The operation isn't valid for this device.
        """
        qubits = operation.qubits
        for qubit in qubits:
            self.validate_qubit(qubit)
        if len(qubits) == 2:
            i = self._aspen_qubit_index(qubits[0])
            j = self._aspen_qubit_index(qubits[1])
            if j not in self.qubit_topology[i]:
                raise UnsupportedRigettiQCSOperation(f'qubits {qubits[0]} and {qubits[1]} do not share an edge')

    def _value_equality_values_(self):
        return self._maximum_qubit_number

    def __repr__(self):
        return f'cirq_rigetti.RigettiQCSAspenDevice(isa={self.isa!r})'

    def _json_dict_(self):
        return {'isa': self.isa.to_dict()}

    @classmethod
    def _from_json_dict_(cls, isa, **kwargs):
        return cls(isa=InstructionSetArchitecture.from_dict(isa))