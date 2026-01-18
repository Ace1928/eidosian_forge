import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def _run_local_calibrations_via_sampler(calibration_requests: Sequence[PhasedFSimCalibrationRequest], sampler: cirq.Sampler):
    """Helper function used by `run_calibrations` to run Local calibrations with a Sampler."""
    return [run_local_xeb_calibration(cast(LocalXEBPhasedFSimCalibrationRequest, calibration_request), sampler) for calibration_request in calibration_requests]