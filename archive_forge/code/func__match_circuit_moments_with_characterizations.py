import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def _match_circuit_moments_with_characterizations(circuit: cirq.Circuit, characterizations: List[PhasedFSimCalibrationResult], gates_translator: Callable[[cirq.Gate], Optional[PhaseCalibratedFSimGate]], merge_subsets: bool, permit_mixed_moments: bool):
    characterized_gate_and_pairs = [(characterization.gate, set(characterization.parameters.keys())) for characterization in characterizations]
    moment_to_calibration: List[Optional[int]] = []
    for moment in circuit:
        pairs_and_gate = _list_moment_pairs_to_characterize(moment, gates_translator, canonicalize_pairs=True, permit_mixed_moments=permit_mixed_moments, sort_pairs=True)
        if pairs_and_gate is None:
            moment_to_calibration.append(None)
            continue
        moment_pairs, moment_gate = pairs_and_gate
        for index, (gate, pairs) in enumerate(characterized_gate_and_pairs):
            if gate == moment_gate and (pairs.issuperset(moment_pairs) if merge_subsets else pairs == set(moment_pairs)):
                moment_to_calibration.append(index)
                break
        else:
            raise ValueError(f'Moment {repr(moment)} of a given circuit is not compatible with any of the characterizations')
    return CircuitWithCalibration(circuit, moment_to_calibration)