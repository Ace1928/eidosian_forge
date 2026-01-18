from typing import Union, Iterable, Tuple
from qiskit.pulse.instructions import Instruction
from qiskit.pulse.schedule import ScheduleBlock, Schedule
from qiskit.pulse.transforms import canonicalization
def _format_schedule_component(sched: Union[InstructionSched, Iterable[InstructionSched]]):
    """A helper function to convert instructions into list of instructions."""
    try:
        sched = list(sched)
        if isinstance(sched[0], int):
            return [tuple(sched)]
        else:
            return sched
    except TypeError:
        return [sched]