import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union
from qiskit import qobj, pulse
from qiskit.assembler.run_config import RunConfig
from qiskit.exceptions import QiskitError
from qiskit.pulse import instructions, transforms, library, schedule, channels
from qiskit.qobj import utils as qobj_utils, converters
from qiskit.qobj.converters.pulse_instruction import ParametricPulseShapes
def _assemble_instructions(sched: Union[pulse.Schedule, pulse.ScheduleBlock], instruction_converter: converters.InstructionToQobjConverter, run_config: RunConfig, user_pulselib: Dict[str, List[complex]]) -> Tuple[List[qobj.PulseQobjInstruction], int]:
    """Assembles the instructions in a schedule into a list of PulseQobjInstructions and returns
    related metadata that will be assembled into the Qobj configuration. Lookup table for
    pulses defined in all experiments are registered in ``user_pulselib``. This object should be
    mutable python dictionary so that items are properly updated after each instruction assemble.
    The dictionary is not returned to avoid redundancy.

    Args:
        sched: Schedule to assemble.
        instruction_converter: A converter instance which can convert PulseInstructions to
                               PulseQobjInstructions.
        run_config: Configuration of the runtime environment.
        user_pulselib: User pulse library from previous schedule.

    Returns:
        A list of converted instructions, the user pulse library dictionary (from pulse name to
        pulse samples), and the maximum number of readout memory slots used by this Schedule.
    """
    sched = transforms.target_qobj_transform(sched)
    max_memory_slot = 0
    qobj_instructions = []
    acquire_instruction_map = defaultdict(list)
    for time, instruction in sched.instructions:
        if isinstance(instruction, instructions.Play):
            if isinstance(instruction.pulse, library.SymbolicPulse):
                is_backend_supported = True
                try:
                    pulse_shape = ParametricPulseShapes.from_instance(instruction.pulse).name
                    if pulse_shape not in run_config.parametric_pulses:
                        is_backend_supported = False
                except ValueError:
                    is_backend_supported = False
                if not is_backend_supported:
                    instruction = instructions.Play(instruction.pulse.get_waveform(), instruction.channel, name=instruction.name)
            if isinstance(instruction.pulse, library.Waveform):
                name = hashlib.sha256(instruction.pulse.samples).hexdigest()
                instruction = instructions.Play(library.Waveform(name=name, samples=instruction.pulse.samples), channel=instruction.channel, name=name)
                user_pulselib[name] = instruction.pulse.samples
        if isinstance(instruction, instructions.Delay) and isinstance(instruction.channel, channels.AcquireChannel):
            continue
        if isinstance(instruction, instructions.Acquire):
            if instruction.mem_slot:
                max_memory_slot = max(max_memory_slot, instruction.mem_slot.index)
            acquire_instruction_map[time, instruction.duration].append(instruction)
            continue
        qobj_instructions.append(instruction_converter(time, instruction))
    if acquire_instruction_map:
        if hasattr(run_config, 'meas_map'):
            _validate_meas_map(acquire_instruction_map, run_config.meas_map)
        for (time, _), instruction_bundle in acquire_instruction_map.items():
            qobj_instructions.append(instruction_converter(time, instruction_bundle))
    return (qobj_instructions, max_memory_slot)