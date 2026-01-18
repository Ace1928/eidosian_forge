from __future__ import annotations
import inspect
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence, Callable
from enum import IntEnum
from typing import Any
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.qobj.converters import QobjToInstructionConverter
from qiskit.qobj.pulse_qobj import PulseQobjInstruction
from qiskit.exceptions import QiskitError
class PulseQobjDef(ScheduleDef):
    """Qobj JSON serialized format instruction sequence.

    A JSON serialized program can be converted into Qiskit Pulse program with
    the provided qobj converter. Because the Qobj JSON doesn't provide signature,
    conversion process occurs when the signature is requested for the first time
    and the generated pulse program is cached for performance.

    .. see_also::
        :class:`.CalibrationEntry` for the purpose of this class.

    """

    def __init__(self, arguments: Sequence[str] | None=None, converter: QobjToInstructionConverter | None=None, name: str | None=None):
        """Define an empty entry.

        Args:
            arguments: User provided argument names for this entry, if parameterized.
            converter: Optional. Qobj to Qiskit converter.
            name: Name of schedule.
        """
        super().__init__(arguments=arguments)
        self._converter = converter or QobjToInstructionConverter(pulse_library=[])
        self._name = name
        self._source: list[PulseQobjInstruction] | None = None

    def _build_schedule(self):
        """Build pulse schedule from cmd-def sequence."""
        schedule = Schedule(name=self._name)
        try:
            for qobj_inst in self._source:
                for qiskit_inst in self._converter._get_sequences(qobj_inst):
                    schedule.insert(qobj_inst.t0, qiskit_inst, inplace=True)
            self._definition = schedule
            self._parse_argument()
        except QiskitError as ex:
            warnings.warn(f'Pulse calibration cannot be built and the entry is ignored: {ex.message}.', UserWarning)
            self._definition = IncompletePulseQobj

    def define(self, definition: list[PulseQobjInstruction], user_provided: bool=False):
        self._source = definition
        self._user_provided = user_provided

    def get_signature(self) -> inspect.Signature:
        if self._definition is None:
            self._build_schedule()
        return super().get_signature()

    def get_schedule(self, *args, **kwargs) -> Schedule | ScheduleBlock | None:
        if self._definition is None:
            self._build_schedule()
        if self._definition is IncompletePulseQobj:
            return None
        return super().get_schedule(*args, **kwargs)

    def __eq__(self, other):
        if isinstance(other, PulseQobjDef):
            return self._source == other._source
        if isinstance(other, ScheduleDef) and self._definition is None:
            self._build_schedule()
        if hasattr(other, '_definition'):
            return self._definition == other._definition
        return False

    def __str__(self):
        if self._definition is None:
            return 'PulseQobj'
        if self._definition is IncompletePulseQobj:
            return 'None'
        return super().__str__()