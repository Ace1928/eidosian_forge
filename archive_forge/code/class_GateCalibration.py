import copy
import pprint
from types import SimpleNamespace
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.qobj.pulse_qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.qobj.common import QobjDictField, QobjHeader
class GateCalibration:
    """Each calibration specifies a unique gate by name, qubits and params, and
    contains the Pulse instructions to implement it."""

    def __init__(self, name, qubits, params, instructions):
        """
        Initialize a single gate calibration. Instructions may reference waveforms which should be
        made available in the pulse_library.

        Args:
            name (str): Gate name.
            qubits (list(int)): Qubits the gate applies to.
            params (list(complex)): Gate parameter values, if any.
            instructions (list(PulseQobjInstruction)): The gate implementation.
        """
        self.name = name
        self.qubits = qubits
        self.params = params
        self.instructions = instructions

    def __hash__(self):
        return hash((self.name, tuple(self.qubits), tuple(self.params), tuple((str(inst) for inst in self.instructions))))

    def to_dict(self):
        """Return a dictionary format representation of the Gate Calibration.

        Returns:
            dict: The dictionary form of the GateCalibration.
        """
        out_dict = copy.copy(self.__dict__)
        out_dict['instructions'] = [x.to_dict() for x in self.instructions]
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new GateCalibration object from a dictionary.

        Args:
            data (dict): A dictionary representing the GateCalibration to create. It
                will be in the same format as output by :func:`to_dict`.

        Returns:
            GateCalibration: The GateCalibration from the input dictionary.
        """
        instructions = data.pop('instructions')
        data['instructions'] = [PulseQobjInstruction.from_dict(x) for x in instructions]
        return cls(**data)