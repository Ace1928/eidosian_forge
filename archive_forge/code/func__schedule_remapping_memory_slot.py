from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING
from qiskit.pulse import channels, exceptions, instructions, utils
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.schedule import Schedule
from qiskit.providers.backend import BackendV2
def _schedule_remapping_memory_slot(schedule: Schedule, qubit_mem_slots: dict[int, int]) -> Schedule:
    """
    A helper function to overwrite MemorySlot index of :class:`.Acquire` instruction.

    Args:
        schedule: A measurement schedule.
        qubit_mem_slots: Mapping of measured qubit index to classical bit index.

    Returns:
        A measurement schedule with new memory slot index.
    """
    new_schedule = Schedule()
    for t0, inst in schedule.instructions:
        if isinstance(inst, instructions.Acquire):
            qubit_index = inst.channel.index
            reg_index = qubit_mem_slots.get(qubit_index, qubit_index)
            new_schedule.insert(t0, instructions.Acquire(inst.duration, channels.AcquireChannel(qubit_index), mem_slot=channels.MemorySlot(reg_index)), inplace=True)
        else:
            new_schedule.insert(t0, inst, inplace=True)
    return new_schedule