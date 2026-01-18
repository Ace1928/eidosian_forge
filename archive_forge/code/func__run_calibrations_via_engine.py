import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def _run_calibrations_via_engine(calibration_requests: Sequence[PhasedFSimCalibrationRequest], processor: AbstractProcessor, max_layers_per_request: int=1, progress_func: Optional[Callable[[int, int], None]]=None):
    """Helper function for run_calibrations.

    This batches and runs calibration requests the normal way: by using engine.run_calibration.
    This function assumes that all inputs have been validated (by `run_calibrations`).
    """
    results = []
    nested_calibration_layers = [[calibration.to_calibration_layer() for calibration in calibration_requests[offset:offset + max_layers_per_request]] for offset in range(0, len(calibration_requests), max_layers_per_request)]
    for cal_layers in nested_calibration_layers:
        job = processor.run_calibration(cal_layers)
        request_results = job.calibration_results()
        results += [calibration.parse_result(result, job) for calibration, result in zip(calibration_requests, request_results)]
        if progress_func:
            progress_func(len(results), len(calibration_requests))
    return results