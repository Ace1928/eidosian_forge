from abc import ABC
from abc import abstractmethod
import datetime
from typing import List, Union, Iterable, Tuple
from qiskit.providers.provider import Provider
from qiskit.providers.models.backendstatus import BackendStatus
from qiskit.circuit.gate import Instruction
class QubitProperties:
    """A representation of the properties of a qubit on a backend.

    This class provides the optional properties that a backend can provide for
    a qubit. These represent the set of qubit properties that Qiskit can
    currently work with if present. However if your backend provides additional
    properties of qubits you should subclass this to add additional custom
    attributes for those custom/additional properties provided by the backend.
    """
    __slots__ = ('t1', 't2', 'frequency')

    def __init__(self, t1=None, t2=None, frequency=None):
        """Create a new :class:`QubitProperties` object.

        Args:
            t1: The T1 time for a qubit in seconds
            t2: The T2 time for a qubit in seconds
            frequency: The frequency of a qubit in Hz
        """
        self.t1 = t1
        self.t2 = t2
        self.frequency = frequency

    def __repr__(self):
        return f'QubitProperties(t1={self.t1}, t2={self.t2}, frequency={self.frequency})'