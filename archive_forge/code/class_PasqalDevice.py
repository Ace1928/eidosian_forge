from typing import Sequence, Any, Union, Dict
import numpy as np
import networkx as nx
import cirq
from cirq import GridQubit, LineQubit
from cirq.ops import NamedQubit
from cirq_pasqal import ThreeDQubit, TwoDQubit, PasqalGateset
@cirq.value.value_equality
class PasqalDevice(cirq.devices.Device):
    """A generic Pasqal device.

    The most general of Pasqal devices, enforcing only restrictions expected to
    be shared by all future devices. Serves as the parent class of all Pasqal
    devices, but can also be used on its own for hosting a nearly unconstrained
    device. When used as a circuit's device, the qubits have to be of the type
    cirq.NamedQubit and assumed to be all connected, the idea behind it being
    that after submission, all optimization and transpilation necessary for its
    execution on the specified device are handled internally by Pasqal.
    """

    def __init__(self, qubits: Sequence[cirq.Qid]) -> None:
        """Initializes a device with some qubits.

        Args:
            qubits (NamedQubit): Qubits on the device, exclusively unrelated to
                a physical position.
        Raises:
            TypeError: If the wrong qubit type is provided.
            ValueError: If the number of qubits is greater than the devices maximum.

        """
        if len(qubits) > 0:
            q_type = type(qubits[0])
        for q in qubits:
            if not isinstance(q, self.supported_qubit_type):
                raise TypeError(f'Unsupported qubit type: {q!r}. This device supports qubit types: {self.supported_qubit_type}')
            if not type(q) is q_type:
                raise TypeError('All qubits must be of same type.')
        if len(qubits) > self.maximum_qubit_number:
            raise ValueError(f'Too many qubits. {type(self)} accepts at most {self.maximum_qubit_number} qubits.')
        self.gateset = PasqalGateset()
        self.qubits = qubits
        self._metadata = cirq.DeviceMetadata(qubits, nx.from_edgelist([(a, b) for a in qubits for b in qubits if a != b]))

    @property
    def supported_qubit_type(self):
        return (NamedQubit,)

    @property
    def maximum_qubit_number(self):
        return 100

    @property
    def metadata(self):
        return self._metadata

    def qubit_list(self):
        return [qubit for qubit in self.qubits]

    def is_pasqal_device_op(self, op: cirq.Operation) -> bool:
        if not isinstance(op, cirq.Operation):
            raise ValueError('Got unknown operation:', op)
        return op in self.gateset

    def validate_operation(self, operation: cirq.Operation):
        """Raises an error if the given operation is invalid on this device.

        Args:
            operation: The operation to validate.

        Raises:
            ValueError: If the operation is not valid.
            NotImplementedError: If the operation is a measurement with an invert
                mask.
        """
        if not isinstance(operation, cirq.GateOperation):
            raise ValueError('Unsupported operation')
        if not self.is_pasqal_device_op(operation):
            raise ValueError(f'{operation.gate!r} is not a supported gate')
        for qub in operation.qubits:
            if not isinstance(qub, self.supported_qubit_type):
                raise ValueError(f'{qub} is not a valid qubit for gate {operation.gate!r}. This device accepts gates on qubits of type: {self.supported_qubit_type}')
            if qub not in self.metadata.qubit_set:
                raise ValueError(f'{qub} is not part of the device.')
        if isinstance(operation.gate, cirq.MeasurementGate):
            if operation.gate.invert_mask != ():
                raise NotImplementedError("Measurements on Pasqal devices don't support invert_mask.")

    def validate_circuit(self, circuit: 'cirq.AbstractCircuit') -> None:
        """Raises an error if the given circuit is invalid on this device.

        A circuit is invalid if any of its moments are invalid or if there
        is a non-empty moment after a moment with a measurement.

        Args:
            circuit: The circuit to validate

        Raises:
            ValueError: If the given circuit can't be run on this device
        """
        super().validate_circuit(circuit)
        has_measurement_occurred = False
        for moment in circuit:
            if has_measurement_occurred:
                if len(moment.operations) > 0:
                    raise ValueError('Non-empty moment after measurement')
            for operation in moment.operations:
                if isinstance(operation.gate, cirq.MeasurementGate):
                    has_measurement_occurred = True

    def __repr__(self):
        return f'pasqal.PasqalDevice(qubits={sorted(self.qubits)!r})'

    def _value_equality_values_(self):
        return self.qubits

    def _json_dict_(self):
        return cirq.protocols.obj_to_dict_helper(self, ['qubits'])