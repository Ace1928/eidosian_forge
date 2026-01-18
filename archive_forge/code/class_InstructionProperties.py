from a backend
from __future__ import annotations
import itertools
from typing import Optional, List, Any
from collections.abc import Mapping
from collections import defaultdict
import datetime
import io
import logging
import inspect
import rustworkx as rx
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.calibration_entries import CalibrationEntry, ScheduleDef
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.pulse.exceptions import PulseError, UnassignedDurationError
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import QubitProperties  # pylint: disable=unused-import
from qiskit.providers.models.backendproperties import BackendProperties
class InstructionProperties:
    """A representation of the properties of a gate implementation.

    This class provides the optional properties that a backend can provide
    about an instruction. These represent the set that the transpiler can
    currently work with if present. However, if your backend provides additional
    properties for instructions you should subclass this to add additional
    custom attributes for those custom/additional properties by the backend.
    """
    __slots__ = ('duration', 'error', '_calibration')

    def __init__(self, duration: float | None=None, error: float | None=None, calibration: Schedule | ScheduleBlock | CalibrationEntry | None=None):
        """Create a new ``InstructionProperties`` object

        Args:
            duration: The duration, in seconds, of the instruction on the
                specified set of qubits
            error: The average error rate for the instruction on the specified
                set of qubits.
            calibration: The pulse representation of the instruction.
        """
        self._calibration: CalibrationEntry | None = None
        self.duration = duration
        self.error = error
        self.calibration = calibration

    @property
    def calibration(self):
        """The pulse representation of the instruction.

        .. note::

            This attribute always returns a Qiskit pulse program, but it is internally
            wrapped by the :class:`.CalibrationEntry` to manage unbound parameters
            and to uniformly handle different data representation,
            for example, un-parsed Pulse Qobj JSON that a backend provider may provide.

            This value can be overridden through the property setter in following manner.
            When you set either :class:`.Schedule` or :class:`.ScheduleBlock` this is
            always treated as a user-defined (custom) calibration and
            the transpiler may automatically attach the calibration data to the output circuit.
            This calibration data may appear in the wire format as an inline calibration,
            which may further update the backend standard instruction set architecture.

            If you are a backend provider who provides a default calibration data
            that is not needed to be attached to the transpiled quantum circuit,
            you can directly set :class:`.CalibrationEntry` instance to this attribute,
            in which you should set :code:`user_provided=False` when you define
            calibration data for the entry. End users can still intentionally utilize
            the calibration data, for example, to run pulse-level simulation of the circuit.
            However, such entry doesn't appear in the wire format, and backend must
            use own definition to compile the circuit down to the execution format.

        """
        if self._calibration is None:
            return None
        return self._calibration.get_schedule()

    @calibration.setter
    def calibration(self, calibration: Schedule | ScheduleBlock | CalibrationEntry):
        if isinstance(calibration, (Schedule, ScheduleBlock)):
            new_entry = ScheduleDef()
            new_entry.define(calibration, user_provided=True)
        else:
            new_entry = calibration
        self._calibration = new_entry

    def __repr__(self):
        return f'InstructionProperties(duration={self.duration}, error={self.error}, calibration={self._calibration})'