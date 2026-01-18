import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def _merge_into_calibrations(calibration: RequestT, calibrations: List[RequestT], pairs_map: Dict[Tuple[Tuple[cirq.Qid, cirq.Qid], ...], int], options: PhasedFSimCalibrationOptions[RequestT]) -> int:
    """Merges a calibration into list of calibrations.

    If calibrations contains an item of which pairs could be expanded to include a new calibration
    pairs, without breaking a moment structure, then those two calibrations will be merged together
    and used as a calibration for both old and newly added calibration.
    If no calibration like that exists, the list will be expanded by the calibration item.

    Args:
        calibration: Calibration to be added.
        calibrations: List of calibrations to be mutated.
        pairs_map: Map from pairs parameter of each calibration on the calibrations list to the
            index on that list. This map will be updated if the calibrations list us updated.
        options: Calibrations options to use when creating a new requests.

    Returns:
        Index of the calibration on the updated calibrations list. If the calibration was added, it
        points to the last element of a list. If not, it points to already existing element.
    """
    new_pairs = set(calibration.pairs)
    for index in pairs_map.values():
        can_merge = calibration.gate == calibrations[index].gate and calibration.options == calibrations[index].options
        if not can_merge:
            continue
        existing_pairs = calibrations[index].pairs
        if new_pairs.issubset(existing_pairs):
            return index
        elif new_pairs.issuperset(existing_pairs):
            calibrations[index] = calibration
            return index
        else:
            new_qubit_pairs = calibration.qubit_to_pair
            existing_qubit_pairs = calibrations[index].qubit_to_pair
            if all((new_qubit_pairs[q] == existing_qubit_pairs[q] for q in set(new_qubit_pairs.keys()).intersection(existing_qubit_pairs.keys()))):
                calibrations[index] = options.create_phased_fsim_request(gate=calibration.gate, pairs=tuple(sorted(new_pairs.union(existing_pairs))))
                return index
    index = len(calibrations)
    calibrations.append(calibration)
    pairs_map[calibration.pairs] = index
    return index