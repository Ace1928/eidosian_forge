import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from qiskit.assembler.run_config import RunConfig
from qiskit.assembler.assemble_schedules import _assemble_instructions as _assemble_schedule
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classicalregister import Clbit
from qiskit.exceptions import QiskitError
from qiskit.qobj import (
from qiskit.utils.parallel import parallel_map
def _extract_common_calibrations(experiments: List[QasmQobjExperiment]) -> Tuple[List[QasmQobjExperiment], Optional[QasmExperimentCalibrations]]:
    """Given a list of ``QasmQobjExperiment``s, each of which may have calibrations in their
    ``config``, collect common calibrations into a global ``QasmExperimentCalibrations``
    and delete them from their local experiments.

    Args:
        experiments: The list of OpenQASM experiments that are being assembled into one qobj

    Returns:
        The input experiments with modified calibrations, and common calibrations, if there
        are any
    """

    def index_calibrations() -> Dict[int, List[Tuple[int, GateCalibration]]]:
        """Map each calibration to all experiments that contain it."""
        exp_indices = defaultdict(list)
        for exp_idx, exp in enumerate(experiments):
            for gate_cal in exp.config.calibrations.gates:
                exp_indices[hash(gate_cal)].append((exp_idx, gate_cal))
        return exp_indices

    def collect_common_calibrations() -> List[GateCalibration]:
        """If a gate calibration appears in all experiments, collect it."""
        common_calibrations = []
        for _, exps_w_cal in exp_indices.items():
            if len(exps_w_cal) == len(experiments):
                _, gate_cal = exps_w_cal[0]
                common_calibrations.append(gate_cal)
        return common_calibrations

    def remove_common_gate_calibrations(exps: List[QasmQobjExperiment]) -> None:
        """For calibrations that appear in all experiments, remove them from the individual
        experiment's ``config.calibrations``."""
        for _, exps_w_cal in exp_indices.items():
            if len(exps_w_cal) == len(exps):
                for exp_idx, gate_cal in exps_w_cal:
                    exps[exp_idx].config.calibrations.gates.remove(gate_cal)
    if not (experiments and all((hasattr(exp.config, 'calibrations') for exp in experiments))):
        return (experiments, None)
    exp_indices = index_calibrations()
    common_calibrations = collect_common_calibrations()
    remove_common_gate_calibrations(experiments)
    for exp in experiments:
        if not exp.config.calibrations.gates:
            del exp.config.calibrations
    return (experiments, QasmExperimentCalibrations(gates=common_calibrations))