from typing import Sequence, Any, Union, Dict
import numpy as np
import networkx as nx
import cirq
from cirq import GridQubit, LineQubit
from cirq.ops import NamedQubit
from cirq_pasqal import ThreeDQubit, TwoDQubit, PasqalGateset
class PasqalVirtualDevice(PasqalDevice):
    """A Pasqal virtual device with qubits in 3d.

    A virtual representation of a Pasqal device, enforcing the constraints
    typically found in a physical device. The qubits can be positioned in 3d
    space, although 2d layouts will be supported sooner and are thus
    recommended. Only accepts qubits with physical placement.
    """

    def __init__(self, control_radius: float, qubits: Sequence[Union[ThreeDQubit, GridQubit, LineQubit]]) -> None:
        """Initializes a device with some qubits.

        Args:
            control_radius: the maximum distance between qubits for a controlled
                gate. Distance is measured in units of the coordinates passed
                into the qubit constructor.
            qubits: Qubits on the device, identified by their x, y, z position.
                Must be of type ThreeDQubit, TwoDQubit, LineQubit or GridQubit.

        Raises:
            ValueError: if the wrong qubit type is provided or if invalid
                parameter is provided for control_radius."""
        super().__init__(qubits)
        if not control_radius >= 0:
            raise ValueError('Control_radius needs to be a non-negative float.')
        if len(self.qubits) > 1:
            if control_radius > 3.0 * self.minimal_distance():
                raise ValueError('Control_radius cannot be larger than 3 times the minimal distance between qubits.')
        self.control_radius = control_radius
        self.gateset = PasqalGateset(include_additional_controlled_ops=False)
        self.controlled_gateset = cirq.Gateset(cirq.AnyIntegerPowerGateFamily(cirq.CZPowGate))

    @property
    def supported_qubit_type(self):
        return (ThreeDQubit, TwoDQubit, GridQubit, LineQubit)

    def validate_operation(self, operation: cirq.Operation):
        """Raises an error if the given operation is invalid on this device.

        Args:
            operation: the operation to validate
        Raises:
            ValueError: If the operation is not valid
        """
        super().validate_operation(operation)
        if operation in self.controlled_gateset:
            for p in operation.qubits:
                for q in operation.qubits:
                    if self.distance(p, q) > self.control_radius:
                        raise ValueError(f'Qubits {p!r}, {q!r} are too far away')

    def validate_moment(self, moment: cirq.Moment):
        """Raises an error if the given moment is invalid on this device.

        Args:
            moment: The moment to validate.
        Raises:
            ValueError: If the given moment is invalid.
        """
        super().validate_moment(moment)
        if len(moment) > 1:
            for operation in moment:
                if not isinstance(operation.gate, cirq.MeasurementGate):
                    raise ValueError('Cannot do simultaneous gates. Use cirq.InsertStrategy.NEW.')

    def minimal_distance(self) -> float:
        """Returns the minimal distance between two qubits in qubits.

        Args:
            qubits: qubit involved in the distance computation

        Raises:
            ValueError: If the device has only one qubit

        Returns:
            The minimal distance between qubits, in spacial coordinate units.
        """
        if len(self.qubits) <= 1:
            raise ValueError('Two qubits to compute a minimal distance.')
        return min([self.distance(q1, q2) for q1 in self.qubits for q2 in self.qubits if q1 != q2])

    def distance(self, p: Any, q: Any) -> float:
        """Returns the distance between two qubits.

        Args:
            p: qubit involved in the distance computation
            q: qubit involved in the distance computation

        Raises:
            ValueError: If p or q not part of the device

        Returns:
            The distance between qubits p and q.
        """
        all_qubits = self.qubit_list()
        if p not in all_qubits or q not in all_qubits:
            raise ValueError('Qubit not part of the device.')
        if isinstance(p, GridQubit):
            return np.sqrt((p.row - q.row) ** 2 + (p.col - q.col) ** 2)
        if isinstance(p, LineQubit):
            return abs(p.x - q.x)
        return np.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2)

    def __repr__(self):
        return f'pasqal.PasqalVirtualDevice(control_radius={self.control_radius!r}, qubits={sorted(self.qubits)!r})'

    def _value_equality_values_(self) -> Any:
        return (self.control_radius, self.qubits)

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.protocols.obj_to_dict_helper(self, ['control_radius', 'qubits'])