import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union
from qiskit import qobj, pulse
from qiskit.assembler.run_config import RunConfig
from qiskit.exceptions import QiskitError
from qiskit.pulse import instructions, transforms, library, schedule, channels
from qiskit.qobj import utils as qobj_utils, converters
from qiskit.qobj.converters.pulse_instruction import ParametricPulseShapes
def _validate_meas_map(instruction_map: Dict[Tuple[int, instructions.Acquire], List[instructions.Acquire]], meas_map: List[List[int]]) -> None:
    """Validate all qubits tied in ``meas_map`` are to be acquired.

    Args:
        instruction_map: A dictionary grouping Acquire instructions according to their start time
                         and duration.
        meas_map: List of groups of qubits that must be acquired together.

    Raises:
        QiskitError: If the instructions do not satisfy the measurement map.
    """
    sorted_inst_map = sorted(instruction_map.items(), key=lambda item: item[0])
    meas_map_sets = [set(m) for m in meas_map]
    for idx, inst in enumerate(sorted_inst_map[:-1]):
        inst_end_time = inst[0][0] + inst[0][1]
        next_inst = sorted_inst_map[idx + 1]
        next_inst_time = next_inst[0][0]
        if next_inst_time < inst_end_time:
            inst_qubits = {inst.channel.index for inst in inst[1]}
            next_inst_qubits = {inst.channel.index for inst in next_inst[1]}
            for meas_set in meas_map_sets:
                common_instr_qubits = inst_qubits.intersection(meas_set)
                common_next = next_inst_qubits.intersection(meas_set)
                if common_instr_qubits and common_next:
                    raise QiskitError('Qubits {} and {} are in the same measurement grouping: {}. They must either be acquired at the same time, or disjointly. Instead, they were acquired at times: {}-{} and {}-{}'.format(common_instr_qubits, common_next, meas_map, inst[0][0], inst_end_time, next_inst_time, next_inst_time + next_inst[0][1]))